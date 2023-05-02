from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from ...rewrites.dead_code_elimination import dce
from ...dialects import toy


# TODO: Migrate to filecheck when dce moves to xdsl
def test_dce():
    @ModuleOp
    @Builder.implicit_region
    def module():
        a = toy.ConstantOp.from_list([1, 2, 3], [3]).res
        b = toy.ConstantOp.from_list([1, 2, 3], [3]).res
        c = toy.AddOp(a, b).res
        d = toy.AddOp(a, c).res
        _e = toy.AddOp(a, d).res
        toy.PrintOp(a)

    @ModuleOp
    @Builder.implicit_region
    def expected():
        a = toy.ConstantOp.from_list([1, 2, 3], [3]).res
        toy.PrintOp(a)

    dce(module)

    assert f"{module}" == f"{expected}"
    assert module.is_structurally_equivalent(expected)
