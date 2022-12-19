from xdsl.frontend.codegen.exception import CodegenException
from xdsl.frontend.program import FrontendProgram


def assert_excepton(p: FrontendProgram, desymref=True):
    try:
        p.compile(desymref)
        assert False, "should not compile"
    except CodegenException as e:
        print(e.msg)
