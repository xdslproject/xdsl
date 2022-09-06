
from dataclasses import dataclass

@dataclass
class Location:
    'Structure definition a location in a file.'
    file: str
    line: int
    col: int
