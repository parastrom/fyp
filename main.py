import os.path
import sys

from dotenv import load_dotenv
import json
from argparse import ArgumentParser
from dataset_loader.spider import Spider



load_dotenv()

DATASETS_PATH = os.getenv('DATASETS')

args = ArgumentParser(description="Conversion of JSON files to custom format")
args.add_argument("--datasets", type=str, default=DATASETS_PATH)
args = args.parse_args()

# Training Datasets
SPIDER = os.path.join(args.datasets, "spider/train_spider.json")
SPARC = os.path.join(args.datasets, "sparc/train.json")
OTHER = os.path.join(args.datasets, "spider/train_others.json")
COSQL = os.path.join(args.datasets, "cosql_dataset/cosql_all_info_dialogs.json")

# Tables
SPIDER_TABLE = os.path.join(args.datasets, "spider/tables.json")
SPARC_TABLE = os.path.join(args.datasets, "sparc/tables.json")
COSQL_TABLE = os.path.join(args.datasets, "cosql_dataset/tables.json")

# Dev set
SPIDER_DEV = os.path.join(args.datasets, "spider/dev.json")
SPARC_DEV = os.path.join(args.datasets, "sparc/dev.json")
#COSQL_DEV = os.path.join(args.datasets, "cosql_dataset/user_intent_prediction/cosql_dev.json")

# Parallel variables
data = []
dbs = []
test_train = []

with open(OTHER) as other, open(SPIDER) as spider, open(SPARC) as sparc,\
    open(COSQL) as cosql, open(SPIDER_DEV) as spider_dev, open(SPARC_DEV) as sparc_dev:

    # Spider
    for x in json.load(spider):
        data.append((x["question"], x["query"], x["db_id"]))
        dbs.append("train_spider")
        test_train.append(1)

    for x in json.load(other):
        data.append((x["question"], x["query"], x["db_id"]))
        dbs.append("train_other")
        test_train.append(1)

    for x in json.load(spider_dev):
        data.append((x["question"], x["query"], x["db_id"]))
        dbs.append("test_spider")
        test_train.append(0)


    #Sparc

    for x in json.load(sparc):
        data.append((x["final"]["utterance"], x["final"]["query"], x["database_id"]))
        dbs.append("train_sparc")
        test_train.append(1)

    for x in json.load(sparc_dev):
        data.append((x["final"]["utterance"], x["final"]["query"], x["database_id"]))
        dbs.append("test_sparc")
        test_train.append(0)


    #CoSQL

    for  k,v in json.load(cosql).items():
        data.append((v["query_goal"], v["sql"], v["db_id"]))
        dbs.append("train_cosql")
        test_train.append(1)

#print(spider_train)

test = Spider()
file = open("items.txt", 'w')
ouput = test._generate_examples(DATASETS_PATH+"spider/train_spider.json", DATASETS_PATH+"spider/database")

with open("items.txt", "w") as fp:
    fp.write('\n'.join("{} {}".format(x[0], x[1]) for x in ouput))
fp.close()


sys.exit()