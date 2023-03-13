import re

def format_sql(input_str: str):
    input_str = input_str.lower()
    for pe in re.findall(r"\w+\s+as\s+t\d+", input_str):


