import json
import logging
import os.path
import pickle
import sys
import attr
import re
import datasets
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Tuple
from torch.utils.data import Dataset
from src.settings import DATASETS_PATH
from pathlib import Path
from tqdm import tqdm
from src.util import linking
import multiprocessing
import gc
import dill
import torch.autograd.grad_mode
from pympler import asizeof

sys.modules['torch.autograd.grad_mode'].F = torch.autograd.grad_mode.no_grad

g_match_score_threshold = 0.3

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
            "column_types": column_types,
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

    def __init__(self, json_example, db, input_encoder):
        super(SpiderExample, self).__init__()

        self.orig = json_example
        self.question = json_example['question']
        self.question_id = json_example['question_id']
        self.columns = db.columns
        self.tables = db.tables
        self.db = db

        self.column_match_cells = self._filter_match_values(json_example['match_values'])

        inputs = input_encoder.encode(self.question, db,
                                            self.column_match_cells)

        self.token_ids = inputs.token_ids
        self.sent_ids = inputs.sent_ids
        self.table_indexes = inputs.table_indexes
        self.column_indexes = inputs.column_indexes
        self.value_indexes = inputs.value_indexes
        self.values = inputs.value_list

        self.token_mapping = inputs.token_mapping
        self.question_tokens = inputs.orig_question_tokens
        self.candi_nums = inputs.candi_nums
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

        schema_links, cell_links = self._linking_wrapper_v2(linking.rasat_schema_linking, linking.rasat_cell_linking)
        q_len = self.column_indexes[0] - 2
        relation_matrix = linking.new_build_relational_matrix(cell_links, schema_links, self.db, q_len)
        return relation_matrix

    def _linking_wrapper_v2(self, fn_schema_linking, fn_cell_linking):
        """Wrapper for schema and cell linking functions."""

        schema_link_res = fn_schema_linking(self.question, self.db)
        cell_link_res = fn_cell_linking(self.question_tokens, self.db)

        schema_result = dict()

        # Convert word id to BERT word pieces id
        reverse_token_mapping = {qid: sublist for sublist in self.token_mapping for qid in sublist}

        for m_name in ["q_col_match", "q_tab_match", "col_q_match", "tab_q_match"]:
            match_matrix = schema_link_res[m_name]
            new_match = dict()
            if m_name in ["q_col_match", "q_tab_match"]:
                for qid, match_types in enumerate(match_matrix):
                    for col_tab_id, match_type in enumerate(match_types):
                        if match_type != "question-table-nomatch" and match_type != "question-column-nomatch":
                            if qid in reverse_token_mapping:
                                for real_qid in reverse_token_mapping[qid]:
                                    new_match[f'{real_qid},{col_tab_id}'] = match_type
            else:  # "col_q_match" and "tab_q_match"
                for col_tab_id, match_types in enumerate(match_matrix):
                    for qid, match_type in enumerate(match_types):
                        if match_type != "table-question-nomatch" and match_type != "column-question-nomatch":
                            if qid in reverse_token_mapping:
                                for real_qid in reverse_token_mapping[qid]:
                                    new_match[f'{col_tab_id},{real_qid}'] = match_type

            schema_result[m_name] = new_match

        # Handle cell_linking results
        cell_result = dict()

        for m_name, matches in cell_link_res.items():
            new_match = {}
            for pos_str, match_type in matches.items():
                qid_str, col_tab_id_str = pos_str.split(',')
                qid, col_tab_id = int(qid_str), int(col_tab_id_str)
                if qid in reverse_token_mapping:
                    for real_qid in reverse_token_mapping[qid]:
                        new_match[f'{real_qid},{col_tab_id}'] = match_type
            cell_result[m_name] = new_match

        return schema_result, cell_result


class SpiderDataset(Dataset):

    def __init__(self, name, input_encoder, label_encoder,
                 is_cached=False, schema_file=None, has_label=True, db_file=None, data_file=None):

        super(SpiderDataset, self).__init__()

        self.name = name
        self.input_encoder = input_encoder
        self.label_encoder = label_encoder
        self.db_schema_file = schema_file
        self.has_label = has_label
        self._qid2index = {}

        spider_data_dict = datasets.load_dataset(path="process/loaders/spider.py", cache_dir=DATASETS_PATH, split=name)
        self.db_path = spider_data_dict["db_path"][0]

        if is_cached:
            self.db_dict, self._examples = None, None
            self.load(db_file, data_file)
        else:
            self.db_dict = process(spider_data_dict)
            self._examples = []
            match_val_name = f'{name}_match_values.json'
            match_values_file = Path(spider_data_dict[0]['db_path']).parent / match_val_name
            train_spider_file = Path(spider_data_dict[0]['data_filepath'])
            if not match_values_file.exists():
                raise FileNotFoundError("match value file not found : " + str(match_values_file))
            with open(match_values_file) as mval_file, open(train_spider_file) as data_file:
                self.collate_examples(json.load(data_file), json.load(mval_file))

    def collate_examples(self, spider_json, match_values, batch_size: int = 32):
        num_processes = max(1, multiprocessing.cpu_count())
        print(num_processes)

        def process_batch(pool, batch_args):
            processed_batch = []
            total = len(batch_args)
            with tqdm(total=total, desc="Processing examples") as pbar:
                for result in pool.imap_unordered(process_single_example, batch_args):
                    processed_batch.append(result)
                    pbar.update(1)
            return processed_batch

        save_dir = Path(DATASETS_PATH / "extracted/af9b2bc9df72ef910b8aa86787e9777f244e799063e643280726daef05d2033e/spider/database")


        with multiprocessing.Pool(num_processes) as pool:  # Move pool creation outside of the process_batch function
            for i in range(0, len(spider_json), batch_size):
                chunk_spider_json = spider_json[i:i + batch_size]
                chunk_match_values = match_values[i:i + batch_size]

                args_list = [
                    (
                        item, m_val, idx,
                        self.db_dict, self.input_encoder, self.label_encoder,
                        self.name, self.has_label
                    )
                    for idx, (item, m_val) in enumerate(zip(chunk_spider_json, chunk_match_values))
                ]

                processed_batch = process_batch(pool, args_list)

                batch_id = i // batch_size
                batch_examples = []
                batch_qid2index = {}

                for result in processed_batch:
                    if result is not None:
                        qid, inputs, outputs = result
                        batch_qid2index[qid] = int(qid[3:])
                        batch_examples.append([inputs, outputs])

                # Save the processed batch using dill
                save_file = save_dir / f'{self.name}_batch_{batch_id}.pkl'
                with open(str(save_file), 'wb') as f:
                    dill.dump([batch_examples, batch_qid2index], f)

    def save(self, save_dir, save_db=True):
        """save data to disk

        Args:
            save_dir (TYPE): NULL
        Raises: NULL
        """
        os.makedirs(save_dir, exist_ok=True)
        if save_db:
            with open(Path(save_dir) / 'db.pkl', 'wb') as ofs:
                dill.dump(self.db_dict, ofs)
        with open(Path(save_dir) / f'{self.name}.pkl', 'wb') as ofs:
            dill.dump([self._examples, self._qid2index], ofs)

    def load(self, db_file, data_file):
        """load data from disk"""
        with open(db_file, 'rb') as ifs:
            unpickler = CustomUnpickler(ifs)
            self.db_dict = unpickler.load()
        with open(data_file, 'rb') as ifs:
            sys.modules['util'] = sys.modules['src.util']
            unpickler = CustomUnpickler(ifs)
            self._examples, self._qid2index = unpickler.load()

    def get_by_qid(self, qid):
        """
        """
        index = self._qid2index[qid]
        return self._examples[index]

    def __getitem__(self, idx):
        """get one example
        """
        return self._examples[idx]

    def __len__(self):
        """size of data examples
        """
        return len(self._examples)


class CustomUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module == 'util':
            module = 'src.util'
        elif module == '__main__':
            if name == 'DB' or name == 'Table' or name == 'Column':
                module = 'src.process.spider_dataset'
        elif module == 'sql_preproc':
            module = 'src.process.sql_preproc'
        elif module == 'text2sql':
            module = 'src'
        return super().find_class(module, name)


def fix_batches(save_dir: str):
    save_dir = Path(save_dir)

    for i, batch_file in enumerate(sorted(save_dir.glob("validation_batch_*.pkl"))):
        # Load the current batch file

        with open(batch_file, 'rb') as f:
            sys.modules['util'] = sys.modules['src.util']
            unpickler = CustomUnpickler(f)
            batch_examples, batch_qid2index = unpickler.load()

            # processed_q = linking.preprocess_question(batch_examples[0][0].question)

        for example in batch_examples:
            preprocessed_q = linking.preprocess_question(example[0].question)
            # assert len(preprocessed_q['processed_question_toks']) == len(example[0].question_tokens)
            q_num = len(preprocessed_q['processed_question_toks'])
            for i in range(0, q_num):
                for j in range(0, q_num):
                    example[0].relations[i, j] = preprocessed_q['relations'][i, j]

        # # Update the qid2index dictionary
        #     updated_qid2index = {qid: int(qid[3:]) for qid in batch_qid2index}
        # Save the updated batch file
        with open(batch_file, 'wb') as f:
            dill.dump([batch_examples, batch_qid2index], f)

def test_loading(cache_dir):

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    db_path = cache_dir / 'test_db_file.pkl'
    val_data_path = cache_dir / 'validation_data_file.pkl'
    train_data_path = cache_dir / 'train_data_file.pkl'
    new_val_path = cache_dir / 'test_train_file.pkl'

    sys.modules['util'] = sys.modules['src.util']
    with open(train_data_path, 'rb') as f:
        train_examples, train_qid2index = dill.load(f)

    with open(val_data_path, 'rb') as f:
        val_examples, val_qid2index = dill.load(f)

    with open(db_path, 'rb') as f:
        dbs = dill.load(f)

    print("helwlo world")
    # with open(val_data_path, 'rb') as f:
    #     val_dbs = pickle.load(f)
    # with open(train_data_path, 'rb') as f:
    #     train_dbs = pickle.load(f)

    # new_dct = dict(train_dbs)
    # new_dct.update(val_dbs)
    #
    # # with open(train_data_path, 'rb') as f:
    # #     examples, qid2index = dill.load(f)
    #
    # with open(new_val_path, 'wb') as f:
    #     pickle.dump(new_dct, f, protocol=pickle.HIGHEST_PROTOCOL)




def collate_batches(save_dir: str):
    save_dir = Path(save_dir)

    examples = []
    qid2index = {}

    sys.modules['util'] = sys.modules['src.util']
    for batch_file in tqdm(sorted(save_dir.glob("batch_*.pkl")), desc="Collating Batches"):
        # Load the current batch file
        with open(batch_file, 'rb') as f:
            unpickler = CustomUnpickler(f)
            batch_examples, batch_qid2index = unpickler.load()

        examples.extend(batch_examples)
        qid2index.update(batch_qid2index)

        gc.collect()

    print(f"Examples size : {asizeof.asizeof(examples)}")
    print(f"qid2index size : {asizeof.asizeof(qid2index)}")
    data_file = save_dir / "train_data_file.pkl"
    with open(data_file, 'wb') as f:
        dill.dump([examples, qid2index], f)

def collate_dbs(cache_dir):

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)

    validation = cache_dir / 'validation_cache.pkl'
    train = cache_dir / "train_cache.pkl"

    with open(validation, 'rb') as val, open(train, 'rb') as tr:
        dev_db = pickle.load(val)
        train_db = pickle.load(tr)

    new_db = dict(train_db)
    new_db.update(dev_db)

    db_save_path = cache_dir / "db_file.pkl"
    with open(db_save_path, 'wb') as f:
        dill.dump(new_db, f)

    print("Hello world")

def process_single_example(args):
    item, m_val, idx, db_dict, input_encoder, label_encoder, name, has_label = args
    db = db_dict[item['db_id']]

    if not input_encoder.check(item, db):
        logging.warning(
            f'check failed: db_id={item["db_id"]}, question = {item["question"]}'
        )
        return None

    if 'question_id' not in item:
        item['question_id'] = f'qid{idx:06d}'
        item['match_values'] = m_val["match_values"]

    inputs = SpiderExample(item, db, input_encoder)
    if 'sql' not in item or not isinstance(item['sql'], dict) or not has_label:
        outputs = None
    else:
        outputs = label_encoder.add_item(name, item['sql'], inputs.values)

    gc.collect()
    return (item['question_id'], inputs, outputs)


if __name__ == '__main__':
    from src.process.bert_encoder import BertInputEncoder
    from src.global_config import get_config
    from sql_preproc import SQLPreproc
    from src.grammar.spider import SpiderLanguage
    from src.settings import ROOT_DIR

    config = get_config()
    model_config = BertInputEncoder(model_config=config.model)
    GrammarClass = SpiderLanguage

    path = ROOT_DIR / Path("conf/spider.asdl")
    label_encoder = SQLPreproc(path,
                               SpiderLanguage,
                               predict_value=config.model.predict_value,
                               is_cached=False)
    spider_ds = SpiderDataset('validation', model_config, label_encoder, is_cached=True)
    spider_ds.save(ROOT_DIR)
    #
    # db_path = ROOT_DIR / Path("conf/cache/db_file.pkl")
    # data_path = ROOT_DIR / Path("conf/cache/train_data_file.pkl")
    # cache_dir = ROOT_DIR / Path("conf/cache")
    #
    # test_loading(cache_dir)

    # # # Call the function with the save directory
    # fix_batches(
    #      '/home/tomisin/Projects/datasets/downloads/extracted/af9b2bc9df72ef910b8aa86787e9777f244e799063e643280726daef05d2033e/spider/database')
