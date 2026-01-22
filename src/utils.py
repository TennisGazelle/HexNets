import json
import pathlib
from typing import List
from tabulate import tabulate 

def table_print(headers: List[str], data: List[List]):
    print(tabulate(data, headers, tablefmt='grid'))

def get_json_file_contents(fileref: pathlib.Path) -> dict:
    if not fileref.exists():
        raise ValueError(f'Expected file {fileref} does not exist')
    with open(fileref, "r") as f:
        return json.load(f)


class Colors:
    """Terminal color codes for colored output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color