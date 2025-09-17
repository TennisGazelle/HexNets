from typing import List
from tabulate import tabulate 

def table_print(headers: List[str], data: List[List]):
    print(tabulate(data, headers, tablefmt='grid'))
