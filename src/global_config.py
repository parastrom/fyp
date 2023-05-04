import json
import _jsonnet as jsonnet
from types import SimpleNamespace
import os
from src.settings import ROOT_DIR
import pathlib


def get_config():

    current_file_path = pathlib.Path(__file__).resolve()
    # Get project root
    project_root = current_file_path.parents[0]

    # Change directory

    os.chdir(project_root)

    path = ROOT_DIR / pathlib.Path('conf/configs.jsonnet')

    config = json.loads(
        jsonnet.evaluate_file(str(path)),
        object_hook=lambda o: SimpleNamespace(**o)
        )

    return config


if __name__ == '__main__':

    config = get_config()

