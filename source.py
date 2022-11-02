import logging
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32, i64, si32, index, Module

p = FrontendProgram(logging.DEBUG)

# a = 5
# with Module():
#     a = 34
#     print(a)
# print(a)

with CodeContext(p, logging.DEBUG):
    x: i32 = 0
    for i in range(10):
        x = 23

