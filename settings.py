import os
from dotenv import load_dotenv

load_dotenv()

DATASETS_PATH = os.getenv('DATASETS')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
