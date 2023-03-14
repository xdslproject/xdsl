from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from enum import Enum, auto


@dataclass(frozen=True)
class Input:
    """
    Used to keep track of the input when parsing.
    """
    content: str = field(repr=False)
    name: str

    @property
    def len(self):
        return len(self.content)

    def __len__(self):
        return self.len

    def get_lines_containing(self,
                             span: Span) -> tuple[list[str], int, int] | None:
        # A pointer to the start of the first line
        start = 0
        line_no = 0
        source = self.content
        while True:
            next_start = source.find('\n', start)
            line_no += 1
            # Handle eof
            if next_start == -1:
                if span.start > len(source):
                    return None
                return [source[start:]], start, line_no
            # As long as the next newline comes before the spans start we can continue
            if next_start < span.start:
                start = next_start + 1
                continue
            # If the whole span is on one line, we are good as well
            if next_start >= span.end:
                return [source[start:next_start]], start, line_no
            while next_start < span.end:
                next_start = source.find('\n', next_start + 1)
            return source[start:next_start].split('\n'), start, line_no

    def at(self, i: int) -> str:
        if i >= self.len:
            raise EOFError()
        return self.content[i]

    def slice(self, start: int, end: int) -> str:
        if end >= self.len:
            raise EOFError()
        return self.content[start:end]


@dataclass(frozen=True)
class Span:
    """
    Parts of the input are always passed around as spans, so we know where they originated.
    """

    start: int
    """
    Start of tokens location in source file, global byte offset in file
    """
    end: int
    """
    End of tokens location in source file, global byte offset in file
    """
    input: Input
    """
    The input being operated on
    """

    def __len__(self):
        return self.len

    @property
    def len(self):
        return self.end - self.start

    @property
    def text(self):
        return self.input.content[self.start:self.end]

    def get_line_col(self) -> tuple[int, int]:
        info = self.input.get_lines_containing(self)
        if info is None:
            return -1, -1
        _lines, offset_of_first_line, line_no = info
        return line_no, self.start - offset_of_first_line

    def print_with_context(self, msg: str | None = None) -> str:
        """
        returns a string containing lines relevant to the span. The Span's contents
        are highlighted by up-carets beneath them (`^`). The message msg is printed
        along these.
        """
        info = self.input.get_lines_containing(self)
        if info is None:
            return "Unknown location of span {}. Error: ".format(msg)
        lines, offset_of_first_line, line_no = info
        # Offset relative to the first line:
        offset = self.start - offset_of_first_line
        remaining_len = max(self.len, 1)
        capture = StringIO()
        print("{}:{}:{}".format(self.input.name, line_no, offset),
              file=capture)
        for line in lines:
            print(line, file=capture)
            if remaining_len < 0:
                continue
            len_on_this_line = min(remaining_len, len(line) - offset)
            remaining_len -= len_on_this_line
            print("{}{}".format(" " * offset, "^" * max(len_on_this_line, 1)),
                  file=capture)
            if msg is not None:
                print("{}{}".format(" " * offset, msg), file=capture)
                msg = None
            offset = 0
        if msg is not None:
            print(msg, file=capture)
        return capture.getvalue()

    def __repr__(self):
        return "{}[{}:{}](text='{}')".format(self.__class__.__name__,
                                             self.start, self.end, self.text)


@dataclass
class Token:

    class Kind(Enum):
        # Markers
        EOF = auto()

        # Identifiers
        BARE_IDENT = auto()
        '''bare-id ::= (letter|[_]) (letter|digit|[_$.])*'''
        AT_IDENT = auto()  # @foo
        '''at-ident ::= `@` (bare-id | string-literal)'''
        HASH_IDENT = auto()  # #foo
        '''hash-ident  ::= `#` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        PERCENT_IDENT = auto()  # %foo
        '''percent-ident  ::= `%` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        CARET_IDENT = auto()  # ^foo
        '''caret-ident  ::= `^` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''
        EXCLAMATION_IDENT = auto()  # !foo
        '''exclamation-ident  ::= `!` (digit+ | (letter|[$._-]) (letter|[$._-]|digit)*)'''

        # Literals
        FLOAT_LIT = auto()  # 1.0
        INTEGER_LIT = auto()  # 1
        STRING_LIT = auto()  # "foo"

        # Punctuation
        ARROW = auto()  # ->
        AT = auto()  # @
        COLON = auto()  # :
        COMMA = auto()  # ,
        ELLIPSIS = auto()  # ...
        EQUAL = auto()  # =
        GREATER = auto()  # >
        L_BRACE = auto()  # {
        L_PAREN = auto()  # (
        L_SQUARE = auto()  # [
        LESS = auto()  # <
        MINUS = auto()  # -
        PLUS = auto()  # +
        QUESTION = auto()  # ?
        R_BRACE = auto()  # }
        R_PAREN = auto()  # )
        R_SQUARE = auto()  # ]
        STAR = auto()  # *
        VERTICAL_BAR = auto()  # |
        FILE_METADATA_BEGIN = auto()  # {-#
        FILE_METADATA_END = auto()  # #-}

    kind: Kind

    span: Span