
from dataclasses import dataclass

@dataclass
class Location:
    'Structure definition a location in a file.'
    file: str
    line: int
    col: int

    def __repr__(self):
        return f'{self.file}:{self.line}:{self.col}'