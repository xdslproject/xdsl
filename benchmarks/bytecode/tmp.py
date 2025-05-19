import sys
import timeit
from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple

from prettytable import PrettyTable

REPEAT = 1000
NUMBER_PER_REPEAT = 1000


############################## dataclass ##############################

empty_dataclass_setup = """
from dataclasses import dataclass

@dataclass
class ExampleDataclass:
    pass
"""

dataclass_create = timeit.repeat(
    stmt="ExampleDataclass()",
    setup=empty_dataclass_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

dataclass_read_setup = """
from dataclasses import dataclass

@dataclass
class ExampleDataclass:
    foo: str

dataclass_instance = ExampleDataclass('some_string_value')
"""

dataclass_read = timeit.repeat(
    stmt="dataclass_instance.foo",
    setup=dataclass_read_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

dataclass_create_with_atr_setup = """
from dataclasses import dataclass

@dataclass
class ExampleDataclass:
    foo: str
"""

dataclass_with_attr = timeit.repeat(
    stmt="ExampleDataclass('some_string_value')",
    setup=dataclass_create_with_atr_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

############################## namedtuple collections ##############################

empty_named_tuple_setup = """
from collections import namedtuple
CollectionsNamedTuple = namedtuple("CollectionsNamedTuple", [])
"""

nt_collections_create = timeit.repeat(
    stmt="CollectionsNamedTuple()",
    setup=empty_named_tuple_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

read_from_named_tuple_setup = """
from collections import namedtuple
CollectionsNamedTuple = namedtuple("CollectionsNamedTuple", ["foo"])
tuple_instance = CollectionsNamedTuple('some_string_value')
"""

nt_collections_read = timeit.repeat(
    stmt="tuple_instance.foo",
    setup=read_from_named_tuple_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

create_namedtuple_with_attr_setup = """
from collections import namedtuple
CollectionsNamedTuple = namedtuple("CollectionsNamedTuple", ["a"])
"""

nt_collections_with_attr = timeit.repeat(
    stmt="CollectionsNamedTuple('some_string_value')",
    setup=create_namedtuple_with_attr_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

############################## NamedTuple typing ##############################

empty_named_tuple_typing_setup = """
from typing import NamedTuple
class TypingNamedTuple(NamedTuple):
    pass
"""

nt_typing_create = timeit.repeat(
    stmt="TypingNamedTuple()",
    setup=empty_named_tuple_typing_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

named_tuple_typing_with_attr_setup = """
from typing import NamedTuple
class TypingNamedTuple(NamedTuple):
    foo: str
"""
nt_typing_with_attr = timeit.repeat(
    stmt="TypingNamedTuple('some_string_value')",
    setup=named_tuple_typing_with_attr_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)

named_tuple_typing_read_setup = """
from typing import NamedTuple
class TypingNamedTuple(NamedTuple):
    foo: str
tuple_instance = TypingNamedTuple('some_string_value')
"""

nt_typing_read = timeit.repeat(
    stmt="tuple_instance.foo",
    setup=named_tuple_typing_read_setup,
    repeat=REPEAT,
    number=NUMBER_PER_REPEAT,
)


############################## Size ##############################


@dataclass
class EmptyDataClass:
    pass


empty_dataclass_size = sys.getsizeof(EmptyDataClass())


class TypingNamedTuple(NamedTuple):
    pass


empty_namedtuple_typing_size = sys.getsizeof(TypingNamedTuple())

EmptyNamedTuple = namedtuple("EmptyNamedTuple", [])

empty_namedtuple_collections_size = sys.getsizeof(EmptyNamedTuple())

############################## Results ##############################


def format_result(result):
    result /= REPEAT  # now result is per one action
    result *= 1_000_000_000  # to nanoseconds
    return f"{result:.2f} ns"


results = PrettyTable(["", "dataclass", "typing NamedTuple", "collections namedtuple"])

results.add_row(
    [
        "Create empty object",
        f"{format_result(min(dataclass_create))}",
        f"{format_result(min(nt_typing_create))}",
        f"{format_result(min(nt_collections_create))}",
    ]
)

results.add_row(
    [
        "Create object with attr",
        f"{format_result(min(dataclass_with_attr))}",
        f"{format_result(min(nt_typing_with_attr))}",
        f"{format_result(min(nt_collections_with_attr))}",
    ]
)
results.add_row(
    [
        "Read attr",
        f"{format_result(min(dataclass_read))}",
        f"{format_result(min(nt_typing_read))}",
        f"{format_result(min(nt_collections_read))}",
    ]
)
results.add_row(
    [
        "empty object size",
        f"{empty_dataclass_size} bytes",
        f"{empty_namedtuple_typing_size} bytes",
        f"{empty_namedtuple_collections_size} bytes",
    ]
)

print(results)
