import json

from src.util import evaluation


def evaluate(dataset, infer_results, output):
    orig_queries = []
    gen_queries = []
    dbs = []

    for i, line in enumerate(infer_results):
        qid, gen_query, _, _ = line.strip().split('\t')
        idx = dataset.get_by_qid(qid)
        example = dataset.__getitem__(idx)
        db_id = example.db.id
        orig_query = example['orig']['query']
        db_path = dataset.db_path + "/" + db_id + "/" + db_id + ".sqlite"

        orig_queries.append(orig_query)
        dbs.append(db_path)
        gen_queries.append(gen_query)

    res = evaluation.evaluate_queries(orig_queries, gen_queries, dbs)
    with open(output / f"eval_{dataset.name}") as fp:
        json.dump(res, fp)