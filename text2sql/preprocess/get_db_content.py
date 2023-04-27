import json
import os
import sqlite3
import datasets
import numpy as np
from tqdm import tqdm


def _get_db_dict_content(db_path):

    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    conn.execute("pragma foreign_keys=ON")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    content_dict = {}
    tables = [row[0] for row in cursor.fetchall()]
    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        column_names = [col[0] for col in cursor.description]
        content_dict[table] = {
            'header': column_names,
            'cell': [[str(cell) for cell in row] for row in rows]
        }

    conn.close()

    return content_dict


def dump_db_json_content(data: datasets.Dataset):

    # All unique DBs by ID and their respective indexes
    db_ids, db_indexes = np.unique(data["db_id"], return_index=True)
    db_path = data["db_path"][0]

    # create dictionary to store content for each table

    for idx, db_id in enumerate(tqdm(db_ids)):
        db_idx = db_indexes[idx]
        sql_db_path = db_path + "/" + db_id + "/" + db_id + ".sqlite"
        db_dct = _get_db_dict_content(sql_db_path)
        json_db_content_path = db_path + "/" + db_id + "/" + db_id + "_content.json"

        with open(json_db_content_path, "w") as outfile:
            json.dump(db_dct, outfile)


if __name__ == "__main__":
    from text2sql.settings import DATASETS_PATH
    train_ds = datasets.load_dataset(path="../dataproc/loaders/spider.py", cache_dir=DATASETS_PATH)
    dump_db_json_content(train_ds['train'])
    dump_db_json_content(train_ds['validation'])
