import json
import datasets
from settings import DATASETS_PATH
from third_party.spider.get_tables import dump_db_json_schema


# TODO Add licensing and citations

class SParC(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sparc",
            version=VERSION,
            description="A dataset for cross-domain Semantic Parsing in Context.",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "utterances": datasets.features.Sequence(datasets.Value("string")),
                "turn_idx": datasets.Value("int32"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
            }
        )

        return datasets.DatasetInfo(
            description="""SParC is a dataset for cross-domain Semantic Parsing in Context. 
            It is the context-dependent/multi-turn version of the Spider task, a complex and cross-domain text-to-SQL challenge. 
            SParC consists of 4,298 coherent question sequences (12k+ unique individual questions annotated with SQL queries annotated by 14 Yale students),
            obtained from user interactions with 200 complex databases over 138 domains.""",
            features=features,
            supervised_keys=None
        )

    def _split_generators(self, dl_manager: None) -> list[datasets.SplitGenerator]:

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": DATASETS_PATH + "sparc/train.json",
                    "db_path": DATASETS_PATH + "sparc/database"
                },
            ),

            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": DATASETS_PATH + "sparc/dev.json",
                    "db_path": DATASETS_PATH + "sparc/database"
                },
            ),
        ]

    def _generate_examples(self, data_filepath, db_path):
        """"Returns the examples in raw text form"""

        with open(data_filepath, encoding="utf-8") as f:
            sparc = json.load(f)

            idx = 0
            for sample in sparc:
                db_id = sample["database_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                    )
                schema = self.schema_cache[db_id]

                db_stuff = {
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [
                        {"column_id": column_id, "other_column_id": other_column_id}
                        for column_id, other_column_id in schema["foreign_keys"]
                    ],
                }

                yield idx, {
                    "utterances": [sample["final"]["utterance"]],
                    "query": sample["final"]["query"],
                    "turn_idx": -1,
                    **db_stuff,
                }

                idx += 1

                utterances = []
                for turn_idx, turn in enumerate(sample["interaction"]):
                    utterances.append(turn["utterance"].strip())
                    yield idx, {
                        "utterances": list(utterances),
                        "query": turn["query"],
                        "turn_idx": turn_idx,
                        **db_stuff
                    }
                    idx += 1
