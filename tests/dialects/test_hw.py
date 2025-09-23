"""
Test HW traits and interfaces.
"""

from unittest.mock import ANY, patch

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    i32,
    i64,
)
from xdsl.dialects.hw import (
    HW,
    Direction,
    DirectionAttr,
    HWModuleLike,
    HWModuleOp,
    InnerRefAttr,
    InnerRefNamespaceTrait,
    InnerRefUserOpInterfaceTrait,
    InnerSymAttr,
    InnerSymbolTableCollection,
    InnerSymbolTableTrait,
    InnerSymPropertiesAttr,
    InnerSymTarget,
    InstanceOp,
    ModulePort,
    ModuleType,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Block
from xdsl.irdl import (
    IRDLOperation,
    Region,
    attr_def,
    irdl_op_definition,
    opt_region_def,
    region_def,
    traits_def,
)
from xdsl.parser import Parser
from xdsl.traits import (
    IsTerminator,
    SingleBlockImplicitTerminator,
    SymbolOpInterface,
    SymbolTable,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


def test_inner_sym_target():
    invalid_target = InnerSymTarget()
    assert not invalid_target

    operand1 = TestOp()

    target = InnerSymTarget(operand1)
    assert target
    assert target.is_op_only()
    assert not target.is_field()

    sub_target = InnerSymTarget.get_target_for_subfield(target, 1)
    assert isinstance(sub_target, InnerSymTarget)
    assert sub_target
    assert not sub_target.is_op_only()
    assert sub_target.is_field()


# Module / Circuit / Wire / Output naming taken from FIRRTL classes using these interfaces.
@irdl_op_definition
class ModuleOp(IRDLOperation):
    name = "module"
    region = region_def()
    sym_name = attr_def(SymbolNameConstraint())
    traits = traits_def(InnerSymbolTableTrait(), SymbolOpInterface())


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "output"
    traits = traits_def(IsTerminator())


@irdl_op_definition
class CircuitOp(IRDLOperation):
    name = "circuit"
    region: Region | None = opt_region_def()
    sym_name = attr_def(SymbolNameConstraint())
    traits = traits_def(
        InnerRefNamespaceTrait(),
        SymbolTable(),
        SingleBlockImplicitTerminator(OutputOp),
    )

    def __post_init__(self):
        for trait in self.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(self, trait)


@irdl_op_definition
class WireOp(IRDLOperation):
    name = "wire"
    sym_name = attr_def(SymbolNameConstraint())
    traits = traits_def(InnerRefUserOpInterfaceTrait())


def test_inner_symbol_table_interface():
    """
    Test operations that conform to InnerSymbolTableTrait
    """
    mod = ModuleOp(
        attributes={"sym_name": StringAttr("symbol_name")}, regions=[[OutputOp()]]
    )

    circ = CircuitOp(attributes={"sym_name": StringAttr("other_name")}, regions=[[mod]])
    circ.verify()

    mod_no_parent = ModuleOp(
        attributes={"sym_name": StringAttr("symbol_name")}, regions=[[OutputOp()]]
    )
    with pytest.raises(
        VerifyException,
        match="Operation module with trait InnerSymbolTableTrait must have a parent with trait SymbolOpInterface",
    ):
        mod_no_parent.verify()

    mod_no_trait_circ = ModuleOp(
        attributes={"sym_name": StringAttr("symbol_name")}, regions=[[OutputOp()]]
    )
    no_trait_circ = TestOp(regions=[[mod_no_trait_circ, OutputOp()]])
    with pytest.raises(
        VerifyException,
        match="Operation module with trait InnerSymbolTableTrait must have a parent with trait InnerRefNamespaceLike",
    ):
        mod_no_trait_circ.verify()
    with pytest.raises(
        VerifyException,
        match="Operation module with trait InnerSymbolTableTrait must have a parent with trait InnerRefNamespaceLike",
    ):
        no_trait_circ.verify()

    @irdl_op_definition
    class MissingTraitModuleOp(IRDLOperation):
        name = "module"
        region = region_def()
        sym_name = attr_def(SymbolNameConstraint())
        traits = traits_def(InnerSymbolTableTrait())

    mod_missing_trait = MissingTraitModuleOp(
        attributes={"sym_name": StringAttr("symbol_name")}, regions=[[OutputOp()]]
    )
    circ_mod_missing_trait = CircuitOp(regions=[[mod_missing_trait, OutputOp()]])
    with pytest.raises(
        VerifyException, match="Operation module must have trait SymbolOpInterface"
    ):
        mod_missing_trait.verify()
    with pytest.raises(
        VerifyException, match="Operation module must have trait SymbolOpInterface"
    ):
        circ_mod_missing_trait.verify()

    @irdl_op_definition
    class MissingAttrModuleOp(IRDLOperation):
        name = "module"
        region = region_def()
        traits = traits_def(InnerSymbolTableTrait(), SymbolOpInterface())

    mod_missing_trait_parent = ModuleOp(regions=[[OutputOp()]])
    MissingAttrModuleOp(regions=[[mod_missing_trait_parent, OutputOp()]])
    with pytest.raises(
        VerifyException,
        match="attribute 'sym_name' expected in operation 'module'",
    ):
        mod_missing_trait_parent.verify()


def test_inner_ref_namespace_interface():
    """
    Test operations that conform to InnerRefNamespaceTrait
    """

    @irdl_op_definition
    class MissingTraitCircuitOp(IRDLOperation):
        name = "circuit"
        region: Region | None = opt_region_def()
        sym_name = attr_def(SymbolNameConstraint())
        traits = traits_def(
            InnerRefNamespaceTrait(), SingleBlockImplicitTerminator(OutputOp)
        )

    wire0 = WireOp(attributes={"sym_name": StringAttr("wire0")})
    mod0 = MissingTraitCircuitOp(
        attributes={"sym_name": StringAttr("mod0")}, regions=[[wire0, OutputOp()]]
    )
    circuit_no_symboltable = CircuitOp(
        attributes={"sym_name": StringAttr("circuit")}, regions=[[mod0]]
    )

    with pytest.raises(
        VerifyException, match="Operation circuit must have trait SymbolTable"
    ):
        circuit_no_symboltable.verify()

    wire1 = WireOp(attributes={"sym_name": StringAttr("wire1")})
    wire2 = WireOp(attributes={"sym_name": StringAttr("wire2")})
    mod1 = ModuleOp(
        attributes={"sym_name": StringAttr("mod1")}, regions=[[wire1, OutputOp()]]
    )
    mod2 = ModuleOp(
        attributes={"sym_name": StringAttr("mod2")}, regions=[[wire2, OutputOp()]]
    )
    circuit = CircuitOp(
        attributes={"sym_name": StringAttr("circuit")}, regions=[[mod1, mod2]]
    )

    # InnerRefUserOpInterfaceTrait.verify_inner_refs() does not do anything, so just mock
    # to check it is called
    with patch.object(
        InnerRefUserOpInterfaceTrait, "verify_inner_refs"
    ) as inner_ref_verif:
        circuit.verify()

    inner_ref_verif.assert_any_call(wire1, ANY)
    inner_ref_verif.assert_any_call(wire2, ANY)
    assert inner_ref_verif.call_count == 2


def test_inner_symbol_table_collection():
    """
    Test operations that pertain to InnerSymbolTableCollection
    """
    wire1 = WireOp(attributes={"sym_name": StringAttr("wire1")})
    wire2 = WireOp(attributes={"sym_name": StringAttr("wire2")})
    mod1 = ModuleOp(
        attributes={"sym_name": StringAttr("mod1")}, regions=[[wire1, OutputOp()]]
    )
    mod2 = ModuleOp(
        attributes={"sym_name": StringAttr("mod2")}, regions=[[wire2, OutputOp()]]
    )
    circuit = CircuitOp(
        attributes={"sym_name": StringAttr("circuit")}, regions=[[mod1, mod2]]
    )

    with pytest.raises(
        VerifyException, match="Operation wire should have InnerRefNamespaceTrait trait"
    ):
        inner_sym_tables = InnerSymbolTableCollection(wire1)

    inner_sym_tables = InnerSymbolTableCollection(circuit)

    with pytest.raises(
        VerifyException, match=r"Trying to insert the same op twice in symbol tables:"
    ):
        inner_sym_tables.populate_and_verify_tables(circuit)

    with pytest.raises(
        VerifyException, match="Operation wire should have InnerSymbolTableTrait trait"
    ):
        inner_sym_tables.get_inner_symbol_table(wire1)

    sym_table1 = inner_sym_tables.get_inner_symbol_table(mod1)
    sym_table2 = inner_sym_tables.get_inner_symbol_table(mod2)
    assert sym_table1 is not sym_table2, (
        "Different InnerSymbolTableTrait objects must return different instances of inner symbol tables"
    )

    unpopulated_inner_sym_tables = InnerSymbolTableCollection()
    sym_table3 = unpopulated_inner_sym_tables.get_inner_symbol_table(mod1)
    sym_table4 = unpopulated_inner_sym_tables.get_inner_symbol_table(mod2)
    assert sym_table3 is not sym_table4, (
        "InnerSymbolTableTrait still behave as expected when created on the fly"
    )


def test_inner_ref_attr():
    """
    Test inner reference attributes
    """
    wire1 = WireOp(attributes={"sym_name": StringAttr("wire1")})
    wire2 = WireOp(attributes={"sym_name": StringAttr("wire2")})
    mod1 = ModuleOp(
        attributes={"sym_name": StringAttr("mod1")}, regions=[[wire1, OutputOp()]]
    )
    mod2 = ModuleOp(
        attributes={"sym_name": StringAttr("mod2")}, regions=[[wire2, OutputOp()]]
    )
    CircuitOp(attributes={"sym_name": StringAttr("circuit")}, regions=[[mod1, mod2]])

    ref = InnerRefAttr("mod2", "wire2")
    assert ref.get_module().data == "mod2", (
        "Name of the referenced module should be returned correctly"
    )


def test_inner_sym_attr():
    """
    Test inner symbol attributes
    """
    invalid_sym_attr = InnerSymAttr()
    assert invalid_sym_attr.get_sym_name() is None, (
        "Invalid InnerSymAttr should return no name"
    )

    sym_attr = InnerSymAttr("sym")
    assert sym_attr.get_sym_name() == StringAttr("sym"), (
        "InnerSymAttr for “ground” type should return name"
    )

    with pytest.raises(VerifyException, match=r"inner symbol cannot have empty name"):
        InnerSymAttr("")

    aggregate_sym_attr = InnerSymAttr(
        [
            InnerSymPropertiesAttr("sym", 0, "public"),
            InnerSymPropertiesAttr("other", 1, "private"),
            InnerSymPropertiesAttr("yet_another", 2, "nested"),
        ]
    )

    assert aggregate_sym_attr.get_sym_name() == StringAttr("sym"), (
        "InnerSymAttr for aggregate types should return name with field ID 0"
    )

    for inner, expected_field_id in zip(aggregate_sym_attr, [0, 1, 2]):
        assert inner.field_id.data == expected_field_id, (
            "InnerSymAttr should allow iterating its properties in order"
        )

    aggregate_without_nested = aggregate_sym_attr.erase(2)
    assert aggregate_without_nested.get_sym_if_exists(2) is None, (
        "InnerSymAttr removal should work"
    )
    assert len(aggregate_without_nested) == 2, (
        "InnerSymAttr removal should correctly change length"
    )


def test_instance_builder():
    MODULE_CTX = """
hw.module @module(in %foo: i32, in %bar: i64, out baz: i32, out qux: i64) {
  hw.output %foo, %bar : i32, i64
}
"""

    ctx = Context()
    ctx.load_dialect(HW)

    module_op = Parser(ctx, MODULE_CTX).parse_module()

    module_op.body.block.add_op(
        inst_op := InstanceOp(
            "test",
            SymbolRefAttr("module"),
            (("foo", create_ssa_value(i32)), ("bar", create_ssa_value(i64))),
            (("baz", i32), ("qux", i64)),
        )
    )

    inst_op.verify()
    assert inst_op.instance_name == StringAttr("test")
    assert inst_op.module_name == SymbolRefAttr("module")
    assert inst_op.arg_names.data == (StringAttr("foo"), StringAttr("bar"))
    assert inst_op.result_names.data == (StringAttr("baz"), StringAttr("qux"))

    assert inst_op.operand_types == (i32, i64)
    assert inst_op.result_types == (i32, i64)


def test_hwmoduleop_hwmodulelike():
    module_type = ModuleType(ArrayAttr(()))

    hw_module = HWModuleOp(
        StringAttr("foo"), module_type, Region((Block((OutputOp(),)),))
    )

    hw_module_like = hw_module.get_trait(HWModuleLike)
    assert hw_module_like is not None
    assert hw_module_like.get_hw_module_type(hw_module) == module_type

    new_module_type = ModuleType(
        ArrayAttr((ModulePort(StringAttr("in1"), i32, DirectionAttr(Direction.INPUT)),))
    )
    hw_module_like.set_hw_module_type(hw_module, new_module_type)
    assert hw_module_like.get_hw_module_type(hw_module) == new_module_type
