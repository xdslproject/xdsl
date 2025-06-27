# RUN: python %s | filecheck %s


from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import CodeContext
from xdsl.frontend.pyast.exception import CodeGenerationException
from xdsl.frontend.pyast.program import FrontendProgram

p = FrontendProgram()
p.register_type(bool, builtin.i1)
p.register_type(int, bigint.bigint)
with CodeContext(p):
    # CHECK: cf.assert %{{.*}}, ""
    def test_assert_I(cond: bool):
        assert cond
        return

    # CHECK: cf.assert %{{.*}}, "some message"
    def test_assert_II(cond: bool):
        assert cond, "some message"
        return


p.compile(desymref=False)
print(p.textual_format())

try:
    with CodeContext(p):
        # CHECK: Expected a string constant for assertion message, found 'ast.Name'
        def test_assert_message_type(cond: bool, a: int):
            assert cond, a
            return

    p.compile(desymref=False)
    print(p.textual_format())
except CodeGenerationException as e:
    print(e.msg)
