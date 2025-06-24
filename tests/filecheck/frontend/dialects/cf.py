# RUN: python %s | filecheck %s

from xdsl.dialects.builtin import I1, I32, IntegerAttr, i1, i32
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(IntegerAttr[I1], i1)
p.register_type(IntegerAttr[I32], i32)
with CodeContext(p):
    # CHECK: cf.assert %{{.*}}, ""
    def test_assert_I(cond: IntegerAttr[I1]):
        assert cond
        return

    # CHECK: cf.assert %{{.*}}, "some message"
    def test_assert_II(cond: IntegerAttr[I1]):
        assert cond, "some message"
        return


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Expected a string constant for assertion message, found 'ast.Name'
        def test_assert_message_type(cond: IntegerAttr[I1], a: IntegerAttr[I32]):
            assert cond, a
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
