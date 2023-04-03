# following
# https://huggingface.co/docs/datasets/v1.12.0/dataset_script.html

import json

import datasets
from settings import DATASETS_PATH
from third_party.spider.get_tables import dump_db_json_schema

# TODO Add licensing and citations

_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataproc for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
"""
_LICENSE = "CC BY-SA 4.0"
_HOMEPAGE = "https://yale-lily.github.io/spider"
_URL = "https://drive.google.com/uc?export=download&confirm=9iBg&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0"


class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="spider",
            version=VERSION,
            description="Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
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
            description="""Spider is a large-scale complex and cross-domain semantic 
            parsing and text-toSQL dataproc annotated by 11 college students""",
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION

        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> list[datasets.SplitGenerator]:

        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/spider/train_spider.json",
                    "db_path": downloaded_filepath + "/spider/database"
                },
            ),

            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/spider/dev.json",
                    "db_path": downloaded_filepath + "/spider/database"
                },
            ),
        ]

    def _generate_examples(self, data_filepath, db_path):
        """Returns the examples in raw text form"""
        with open(data_filepath, encoding="utf-8") as f:
            spider = json.load(f)

            for idx, sample in enumerate(spider):
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                    )
                schema = self.schema_cache[db_id]
                yield idx, {
                    "query": sample["query"],
                    "question": sample["question"],
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
