# RUN: python %s | filecheck %s


from xdsl.dialects import bigint, builtin
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException

ctx = PyASTContext(post_transforms=[])
ctx.register_type(bool, builtin.i1)
ctx.register_type(int, bigint.bigint)


# CHECK: cf.assert %{{.*}}, ""
@ctx.parse_program
def test_assert_I(cond: bool):
    assert cond
    return


print(test_assert_I.module)


# CHECK: cf.assert %{{.*}}, "some message"
@ctx.parse_program
def test_assert_II(cond: bool):
    assert cond, "some message"
    return


print(test_assert_II.module)


# CHECK: Expected a string constant for assertion message, found 'ast.Name'
@ctx.parse_program
def test_assert_message_type(cond: bool, a: int):
    assert cond, a
    return


try:
    test_assert_message_type.module
except CodeGenerationException as e:
    print(e.msg)
