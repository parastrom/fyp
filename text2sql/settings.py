import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

DATASETS_PATH = os.getenv('DATASETS')
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
