# RUN: python %s | filecheck %s

from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i32
from xdsl.frontend.exception import CodeGenerationException
from xdsl.frontend.program import FrontendProgram

p = FrontendProgram()
with CodeContext(p):
    # CHECK: "cf.assert"(%{{.*}}) <{"msg" = ""}> : (i1) -> ()
    def test_assert_I(cond: i1):
        assert cond
        return

    # CHECK: "cf.assert"(%{{.*}}) <{"msg" = "some message"}> : (i1) -> ()
    def test_assert_II(cond: i1):
        assert cond, "some message"
        return


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Expected a string constant for assertion message, found 'ast.Name'
        def test_assert_message_type(cond: i1, a: i32):
            assert cond, a
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
