import datasets
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dataproc import spider_dataset as dutils

from settings import DATASETS_PATH

# args = ArgumentParser(description="Conversion of JSON files to custom format")
# args.add_argument("--datasets", type=str, default=DATASETS_PATH)
# args = args.parse_args()
#
# # Training Datasets
# SPIDER = os.path.join(args.datasets, "spider/train_spider.json")
# SPARC = os.path.join(args.datasets, "sparc/train.json")
# OTHER = os.path.join(args.datasets, "spider/train_others.json")
# COSQL = os.path.join(args.datasets, "cosql_dataset/cosql_all_info_dialogs.json")
#
# # Tables
# SPIDER_TABLE = os.path.join(args.datasets, "spider/tables.json")
# SPARC_TABLE = os.path.join(args.datasets, "sparc/tables.json")
# COSQL_TABLE = os.path.join(args.datasets, "cosql_dataset/tables.json")
#
# # Dev set
# SPIDER_DEV = os.path.join(args.datasets, "spider/dev.json")
# SPARC_DEV = os.path.join(args.datasets, "sparc/dev.json")
# #COSQL_DEV = os.path.join(args.datasets, "cosql_dataset/user_intent_prediction/cosql_dev.json")
#
# # Parallel variables
# data = []
# dbs = []
# test_train = []
#
# with open(OTHER) as other, open(SPIDER) as spider, open(SPARC) as sparc,\
#     open(COSQL) as cosql, open(SPIDER_DEV) as spider_dev, open(SPARC_DEV) as sparc_dev:
#
#     # Spider
#     for x in json.load(spider):
#         data.append((x["question"], x["query"], x["db_id"]))
#         dbs.append("train_spider")
#         test_train.append(1)
#
#     for x in json.load(other):
#         data.append((x["question"], x["query"], x["db_id"]))
#         dbs.append("train_other")
#         test_train.append(1)
#
#     for x in json.load(spider_dev):
#         data.append((x["question"], x["query"], x["db_id"]))
#         dbs.append("test_spider")
#         test_train.append(0)
#
#
#     #Sparc
#
#     for x in json.load(sparc):
#         data.append((x["final"]["utterance"], x["final"]["query"], x["database_id"]))
#         dbs.append("train_sparc")
#         test_train.append(1)
#
#     for x in json.load(sparc_dev):
#         data.append((x["final"]["utterance"], x["final"]["query"], x["database_id"]))
#         dbs.append("test_sparc")
#         test_train.append(0)
#
#
#     #CoSQL
#
#     for  k,v in json.load(cosql).items():
#         data.append((v["query_goal"], v["sql"], v["db_id"]))
#         dbs.append("train_cosql")
#         test_train.append(1)
#
# #print(spider_train)

test = datasets.load_dataset(path="dataproc/loaders/spider.py", cache_dir=DATASETS_PATH, split='train')
# primary = test["db_primary_keys"]
# cols = test["db_column_names"]
#table_names = test["db_table_names"]
# db_ids = test["db_id"]

# db_ids, db_indexes = np.unique(test["db_id"], return_index=True)
#
# # unique_db_idxs = test.unique("db_id")
#
# db_graphs = dict()

# for idx, db_id in enumerate(db_ids):
#     db_idx = db_indexes[idx]
#
#     columns = test["db_column_names"][db_idx]
#     table_names = test["db_table_names"][db_idx]
#     column_types = test["db_column_types"][db_idx]
#     foreign_keys = test["db_foreign_keys"][db_idx]
#     primary_keys = test["db_primary_keys"][db_idx]
#
#     g = nx.Graph()
#
#     for i, sample in enumerate(zip(columns["table_id"][1:], columns["column_name"][1:])):
#         table_idx = sample[0]
#         table = table_names[table_idx]
#         col_name = sample[1].replace(" ", "_")
#         g.add_node(
#             i+1, id=f"{table}.{col_name}", name=col_name, table=table,
#             primary=True if i+1 in primary_keys["column_id"] else False,
#             type=column_types[i+1]
#         )
#
#     for (n1, n2) in zip(foreign_keys["column_id"], foreign_keys["other_column_id"]):
#         g.add_edge(n1, n2, foreign=True)
#
#     if idx > 2:
#         break

# db_idx = 0
#
# columns = test["db_column_names"][db_idx]
# table_names = test["db_table_names"][db_idx]
# column_types = test["db_column_types"][db_idx]
# foreign_keys = test["db_foreign_keys"][db_idx]
# primary_keys = test["db_primary_keys"][db_idx]
#
# g = nx.Graph()
#
# for i, sample in enumerate(zip(columns["table_id"][1:], columns["column_name"][1:])):
#     table_idx = sample[0]
#     table = table_names[table_idx]
#     col_name = sample[1].replace(" ", "_")
#     g.add_node(
#         i+1, id=f"{table}.{col_name}", name=col_name, table=table,
#         primary=True if i+1 in primary_keys["column_id"] else False,
#         type=column_types[i+1]
#     )
#
#     for (n1, n2) in zip(foreign_keys["column_id"], foreign_keys["other_column_id"]):
#         g.add_edge(n1-1, n2-1, foreign=True)
#
# plt.figure(figsize=(10, 10), dpi=150)
# nx.draw(g, node_size=60, with_labels=True,)
# plt.savefig("Graph.png", format="PNG")
# plt.show()
# counter = 0
# for i, sample in enumerate(cols):
#     db_id = test["db_id"][i]
#     if db_id not in db_graphs:
#
#         tables = test["db_table_names"][i]
#         col_types = test["db_column_types"][i]
#
#         db_graphs[db_id] = i
#         print("{}", counter)
#         counter += 1
# def graph_gen(dt: datasets.Dataset):

schemas = dutils.process(test)
