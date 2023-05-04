import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

DATASETS_PATH = os.getenv('DATASETS')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
