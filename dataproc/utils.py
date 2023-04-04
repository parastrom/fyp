import json

import attr
import re
import datasets
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from settings import DATASETS_PATH
from fuzzywuzzy import fuzz
from tqdm import tqdm

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
    cells = attr.ib(factory=list)
    foreign_key = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)
    primary_keys_id = attr.ib(factory=list)
    foreign_key_tables = attr.ib(factory=set)


@attr.s
class Schema:
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

        schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema)

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

        self.columns_match_cells = self._filter_match_value(json_example['match_values'])


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





    def _generate_match_values(self, data: datasets.Dataset):

        min_match_score = 75
        match_values_dict = {}

        for dict in tqdm(data['db_column_names']):
            column_name = dict['column_name']
            table_name = data[dict['table_id']]

            column_id = f"{table_name.lower()}_{column_name.lower()}"

