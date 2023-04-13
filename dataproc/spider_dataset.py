import json
import logging
import os.path

import attr
import re
import datasets
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Tuple
from torch.utils.data import Dataset
from settings import DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from util import linking


g_match_score_threshold = 0.3
@attr.s
class Item:
    text: str = attr.ib()
    code: str = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    processed_name = attr.ib(default=None)
    processed_toks = attr.ib(default=None)
    cells = attr.ib(factory=list)
    foreign_key = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    processed_name = attr.ib(default=None)
    processed_toks = attr.ib(default=None)
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)
    primary_keys_id = attr.ib(factory=list)
    foreign_key_tables = attr.ib(factory=set)


@attr.s
class DB:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()
    connections = attr.ib(default=None)


def postprocess_original_name(s: str):
    return re.sub(r'([A-Z]+)', r' \1', s).replace('_', ' ').lower().strip()


def _extract_column_cells(table_names, db_content):

    column_cells = [table_names]

    for table_name in table_names:
        table_info = db_content.get(table_name, None)
        if table_info is None:
            return None

        rows = table_info.get('cell', [])
        if len(rows) == 0:
            rows = [[] for _ in db_content[table_name]['header']]
            column_cells.extend(rows)
        else:
            column_cells.extend(list(zip(*rows)))

    return column_cells


def process(data: datasets.Dataset):
    schemas = {}

    # Get db names and the index of their first appearance
    db_ids, db_indexes = np.unique(data["db_id"], return_index=True)
    db_path = data["db_path"][0]

    for idx, db_id in enumerate(tqdm(db_ids)):

        db_idx = db_indexes[idx]

        db_columns = data["db_column_names"][db_idx]
        table_names = data["db_table_names"][db_idx]
        column_types = data["db_column_types"][db_idx]
        foreign_keys = data["db_foreign_keys"][db_idx]
        primary_keys = data["db_primary_keys"][db_idx]
        json_db_content_path = db_path + "/" + db_id + "/" + db_id + "_content.json"

        db_content = {}

        with open(json_db_content_path, "r") as json_file:
            db_content = json.load(json_file)

        column_cells = _extract_column_cells(table_names, db_content)

        if column_cells is None:
            column_cells = [[] for _ in db_columns['column_name']]
        assert len(column_cells) == len(db_columns['column_name'])

        schema = {
            "db_id": db_id,
            "table_names": table_names,
            "column_names": db_columns,
            "column_types":  column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,

        }

        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                orig_name=name,
            )
            for i, name in enumerate(table_names)
        )

        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                orig_name=col_name,
                type=col_type,
                cells=[
                    x for x in set([str(c) for c in column_cells[i]])
                    if len(x) <= 20 or x.startswith('item_')
                ],
            )
            for i, (table_id, col_name, col_type) in enumerate(zip(
                db_columns["table_id"],
                db_columns["column_name"],
                column_types
            ))
        )

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for col_id in primary_keys["column_id"]:

            # Add primary keys
            column = columns[col_id]
            column.table.primary_keys.append(column)
            column.table.primary_keys_id.append(col_id)

        foreign_key_graph = nx.DiGraph()
        for (source_col_id, dest_col_id) in zip(foreign_keys["column_id"], foreign_keys["other_column_id"]):

            # Add foreign key
            source_col = columns[source_col_id]
            dest_col = columns[dest_col_id]
            source_col.foreign_key = dest_col
            columns[source_col_id].table.foreign_key_tables.add(dest_col_id)
            foreign_key_graph.add_edge(
                source_col.table.id,
                dest_col.table.id,
                columns=(source_col_id, dest_col_id)
            )
            foreign_key_graph.add_edge(
                dest_col.table.id,
                source_col.table.id,
                columns=(dest_col_id, source_col_id)
            )

        assert db_id not in schemas

        schemas[db_id] = DB(db_id, tables, columns, foreign_key_graph, schema)

    return schemas


class SpiderExample(object):
    """Processing Methods"""

    def __init__(self, json_example, db,  input_encoder):
        super(SpiderExample, self).__init__()

        self.orig = json_example
        self.question = json_example['question']
        self.question_id = json_example['question_id']
        self.columns = db.columns
        self.tables = db.tables
        self.db = db

        self.column_match_cells = self._filter_match_value(json_example['match_values'])

        ernie_inputs = input_encoder.encode(self.question, db,
                                            self.column_match_cells)

        self.token_ids = ernie_inputs.token_ids
        self.sent_ids = ernie_inputs.sent_ids
        self.table_indexes = ernie_inputs.table_indexes
        self.column_indexes = ernie_inputs.column_indexes
        self.value_indexes = ernie_inputs.value_indexes
        self.values = ernie_inputs.value_list

        self.token_mapping = ernie_inputs.token_mapping
        self.question_tokens = ernie_inputs.orig_question_tokens
        self.candi_nums = ernie_inputs.candi_nums
        self.relations = self._compute_relations()

    def _filter_match_values(self, match_values_info):
        lst_result = []
        for column_values in match_values_info:
            filtered_results = []
            for value, score in column_values:
                if score > g_match_score_threshold:
                    filtered_results.append(value)
                else:
                    break
            lst_result.append(filtered_results)
        return lst_result

    def _compute_relations(self):
        schema_links = self._linking_wrapper(linking.compute_schema_linking())
        cell_value_links = self._linking_wrapper(linking.compute_cell_value_linking())
        link_info_dict = {
            'sc_link': schema_links,
            'cv_link': cell_value_links,
        }

        q_len = self.column_indexes[0] - 2
        c_len = len(self.columns)
        t_len = len(self.tables)
        total_len = q_len + c_len + t_len
        relation_matrix = linking.build_relation_matrix(
            link_info_dict, total_len, q_len, c_len,
            list(range(c_len + 1)), list(range(t_len + 1)), self.db)
        return relation_matrix

    def _linking_wrapper(self, fn_linking):
        """wrapper for linking function, do linking and id convert
        """
        link_result = fn_linking(self.question, self.db)

        # convert words id to BERT word pieces id
        new_result = {}
        for m_name, matches in link_result.items():
            new_match = {}
            for pos_str, match_type in matches.items():
                qid_str, col_tab_id_str = pos_str.split(',')
                qid, col_tab_id = int(qid_str), int(col_tab_id_str)
                for real_qid in self.token_mapping[qid]:
                    new_match[f'{real_qid},{col_tab_id}'] = match_type
            new_result[m_name] = new_match
        return new_result



class SpiderDataset(Dataset):

    def __int__(self, name, db_file, data_file, input_encoder, label_encoder,
                is_cached=False, schema_file=None, has_label=True):

        super(SpiderDataset, self).__init__()

        self.name = name
        self.input_encoder = input_encoder
        self.label_encoder = label_encoder
        self.db_schema_file = schema_file
        self.has_label = has_label
        self._qid2index = {}

        spider_data_dict = datasets.load_dataset(path="dataproc/loaders/spider.py", cache_dir=DATASETS_PATH)

        self.db_dict = process(spider_data_dict['train'])
        self._examples = []
        match_values_file = Path(spider_data_dict[0]['db_path']).parent / 'match_values.json'
        train_spider_file = Path(spider_data_dict[0]['data_filepath'])
        if not match_values_file.exists():
            raise FileNotFoundError("match value file not found : "+str(match_values_file))
        with open(match_values_file) as mval_file, open(train_spider_file) as data_file:
            self.collate_examples(json.load(data_file), json.load(mval_file))

    def collate_examples(self, spider_json: List[dict], match_values: List[dict]):

        for idx, (item, m_val) in tqdm(enumerate(zip(spider_json, match_values))):
            db = self.db_dict[item['db_id']]

            if not self.input_encoder.check(item, db):
                logging.warning(
                    f'check failed: db_id={item["db_id"]}, question = {item["question"]}'
                )
                continue

            if 'question_id' not in item:
                item['question_id'] = f'qid{idx:06d}'
                item['match_values'] = m_val["match_values"]

            inputs = SpiderExample(item, db, self.input_encoder)
            if 'sql' not in item or not isinstance(item['sql'], dict) or not self.has_label:
                outputs = None
            else:
                outputs = self.label_encoder.add_item(self.name, item['sql'], inputs.values)
            self._qid2index[item['question_id']] = len(self._examples)
            self._examples.append([inputs, outputs])



