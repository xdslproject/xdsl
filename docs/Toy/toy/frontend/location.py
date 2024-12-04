from dataclasses import dataclass
from pathlib import Path


@dataclass
class Location:
    "Structure definition a location in a file."

    file: Path
    line: int
    col: int

    def __repr__(self):
        return f"{self.file}:{self.line}:{self.col}"
