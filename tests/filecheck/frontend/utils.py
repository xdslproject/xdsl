# RUN: python %s
# TODO: This results in PASS for utils.py file when running lit tests, which is ugly.
from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.program import FrontendProgram


def assert_excepton(p: FrontendProgram, desymref=True):
    try:
        p.compile(desymref)
        assert False, "should not compile"
    except CodegenException as e:
        print(e.msg)
