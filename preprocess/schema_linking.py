import json
import os.path
import sys

# setting path
sys.path.append('../dataproc')

from dataproc import spider_dataset
import datasets
from settings import DATASETS_PATH
from collections import defaultdict


def _build(cells):
    dct_index = defaultdict(set)
    for cell in set(cells):
        if type(cell) is not str:
            continue
        cell = cell.strip()
        cell_chars = tuple(list(cell.lower()))
        dct_index[cell.lower()].add((cell, len(cell_chars)))
        for pos in range(len(cell_chars) -1):
            bigram = cell_chars[pos:pos+2]
            dct_index[bigram].add((cell, len(cell_chars) - 1))
    return dct_index


def build_cell_index(db_dict):
    for db in db_dict.values():
        column_cells = []
        for column in db.columns:
            cell_index = _build(column.cells)
            column_cells.append(cell_index)
        db.column_cells_index = column_cells


def search_values(query, db, extra_values):
    lst_match_values = []
    for column, cell_index in zip(db.columns, db.column_cells_index):
        if column.id == 0:
            lst_match_values.append([])
            continue

        candi_cnt = defaultdict(float)
        query_chars = tuple(list(query.lower()))
        appear_set = set()
        for pos in range(len(query_chars)):
            unigram = query_chars[pos]
            if len(unigram) > 2 and unigram not in appear_set and unigram in cell_index:
                for cell, base in cell_index[unigram]:
                    candi_cnt[cell] += 1.0 / base
            if pos == len(query_chars) - 1:
                break

            bigram = query_chars[pos:pos+2]
            if bigram not in cell_index:
                continue
            if bigram in appear_set:
                continue
            appear_set.add(bigram)
            for cell, base in cell_index[bigram]:
                candi_cnt[cell] += 1.0 / base

        if extra_values is not None and column.id in extra_values:
            gold_values = extra_values[column.id]
            for gval in gold_values:
                candi_cnt[str(gval)] += 2.0

        lst_match_values.append(list(
            sorted(
                candi_cnt.items(), key=lambda x: x[1], reverse=True
            )
        )[:10])

    return lst_match_values


def extract_value_from_sql(sql_json):
    dct_col_values = defaultdict(list)

    def _merge_dict(base_dict, extra_dict):
        for k, v in extra_dict.items():
            base_dict[k].extend(v)

    def _extract_value_from_sql_cond(cond, dct_col_values):
        if isinstance(cond[3], dict):
            _merge_dict(dct_col_values, extract_value_from_sql(cond[3]))
            return
        col_id = cond[2][1][1]
        dct_col_values[col_id].append(cond[3])
        if cond[4] is not None:
            dct_col_values[col_id].append(cond[4])

    for table_unit in sql_json['from']['table_units']:
        if isinstance(table_unit[1], dict):
            _merge_dict(dct_col_values, extract_value_from_sql(table_unit[1]))

    for key in ['where', 'having']:
        for cond in sql_json[key][::2]:
            _extract_value_from_sql_cond(cond, dct_col_values)

    for key in ['intersect', 'union', 'except']:
        if sql_json[key] is not None:
            _merge_dict(dct_col_values, extract_value_from_sql(sql_json[key]))

    return dct_col_values


if __name__ == "__main__":

    db_dct = datasets.load_dataset(path="../dataproc/loaders/spider.py", cache_dir=DATASETS_PATH, split='train')
    lst_output = []
    dbs = utils.process(db_dct)
    build_cell_index(dbs)
    fpath = db_dct["data_filepath"][0]
    with open(fpath, "r") as spider_json:
        data = json.load(spider_json)

    for idx, sample in enumerate(db_dct):

        question = sample['question']
        question_id = f'qid{idx:06d}'
        db_id = sample['db_id']
        db = dbs[db_id]
        sql_json = data[idx]['sql']
        extra_values = extract_value_from_sql(sql_json)

        match_values = search_values(question, db, extra_values)
        lst_output.append({
            "question_id": question_id,
            "question": question,
            "db_id": db_id,
            "match_values": match_values
        })

    parent_dir = os.path.dirname(db_dct['db_path'][0])
    outfile_path = parent_dir + "/" + "match_values.json"

    with open(outfile_path, "w") as out_file:
        json.dump(lst_output, out_file, indent=2, ensure_ascii=False)

