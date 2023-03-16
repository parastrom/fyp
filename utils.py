# data preprocessing

import re
import numpy as np
import pandas as pd
import dgl
import datasets as dt
import networkx as nx
from torch.utils.data import Dataset

def db_to_graph(dataset: dt.Dataset):
    """
    At any given moment we cannot isolate a specific database due
    to the way we loaded in our data, therefore we take our entire
    dataset and construct a dictionary with the db_id as key and
    the graph as the value.

    Convert the given database to a DGL graph\n
    DB features:\n
    **db_id**: datasets.Value("string")\n
    **db_path**: datasets.Value("string")\n
    **db_table_names**: datasets.features.Sequence
        - datasets.Value("string")\n
    **db_column_names**: datasets.features.Sequence
        - table_id": datasets.Value("int32")\n
        - column_name": datasets.Value("string"\n
    **db_column_types**: datasets.features.Sequence
        - column_id": datasets.Value("string")\n
    **db_primary_keys**: datasets.features.Sequence
        - column_id": datasets.Value("int32")\n
    **db_foreign_keys**: datasets.features.Sequence
        - "column_id": datasets.Value("int32"),
        - "other_column_id": datasets.Value("int32"),
    :param dataset: Dataset
    :return: dgl graph
    """

    columns = dataset["db_column_names"]
    column_types = dataset["db_column_types"]
    table_names = dataset["db_table_names"]
    primary_keys = dataset["db_primary_keys"]
    foreign_keys = dataset["db_foreign_keys"]

    g = nx.Graph()

    col_count = 0
    cols = []
    for sample in columns:



    for i,c in enumerate(columns):
        # c : Dictionary with table_id and col_name

        col_name = c["column_name"].replace(" ", "_")
        table_name = table_names[c["table_id"]]

        for keys in primary_keys:
            if keys[

        g.add_node(
            i, id=f"{table_name}.{col_name}",col_name=col_name, table_name=table_name,
            primary = True if next((key for key in primary_keys if key["column_id"] == i), None) is not None else False

        )
