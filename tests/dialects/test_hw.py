"""
Test HW traits and interfaces.
"""

from unittest.mock import ANY, patch

import pytest

from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.hw import (
    InnerRefAttr,
    InnerRefNamespaceTrait,
    InnerRefUserOpInterfaceTrait,
    InnerSymAttr,
    InnerSymbolTableCollection,
    InnerSymbolTableTrait,
    InnerSymPropertiesAttr,
    InnerSymTarget,
)
from xdsl.dialects.test import TestOp
from xdsl.irdl import (
    IRDLOperation,
    Region,
    attr_def,
    irdl_op_definition,
    opt_region_def,
    region_def,
)
from xdsl.traits import (
    IsTerminator,
    SingleBlockImplicitTerminator,
    SymbolOpInterface,
    SymbolTable,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException


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
    sym_name = attr_def(StringAttr)
    traits = frozenset({InnerSymbolTableTrait(), SymbolOpInterface()})


@irdl_op_definition
class OutputOp(IRDLOperation):
    name = "output"
    traits = frozenset({IsTerminator()})


@irdl_op_definition
class CircuitOp(IRDLOperation):
    name = "circuit"
    region: Region | None = opt_region_def()
    sym_name = attr_def(StringAttr)
    traits = frozenset(
        {
            InnerRefNamespaceTrait(),
            SymbolTable(),
            SingleBlockImplicitTerminator(OutputOp),
        }
    )

    def __post_init__(self):
        for trait in self.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(self, trait)


@irdl_op_definition
class WireOp(IRDLOperation):
    name = "wire"
    sym_name = attr_def(StringAttr)
    traits = frozenset({InnerRefUserOpInterfaceTrait()})


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
        match="Operation module with trait InnerSymbolTableTrait must have a parent with trait InnerRefNamespaceTrait",
    ):
        mod_no_trait_circ.verify()
    with pytest.raises(
        VerifyException,
        match="Operation module with trait InnerSymbolTableTrait must have a parent with trait InnerRefNamespaceTrait",
    ):
        no_trait_circ.verify()

    @irdl_op_definition
    class MissingTraitModuleOp(IRDLOperation):
        name = "module"
        region = region_def()
        sym_name = attr_def(StringAttr)
        traits = frozenset({InnerSymbolTableTrait()})

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
        traits = frozenset({InnerSymbolTableTrait(), SymbolOpInterface()})

    mod_missing_trait_parent = ModuleOp(regions=[[OutputOp()]])
    MissingAttrModuleOp(regions=[[mod_missing_trait_parent, OutputOp()]])
    with pytest.raises(
        VerifyException,
        match="attribute sym_name expected",
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
        sym_name = attr_def(StringAttr)
        traits = frozenset(
            {InnerRefNamespaceTrait(), SingleBlockImplicitTerminator(OutputOp)}
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
    assert (
        sym_table1 is not sym_table2
    ), "Different InnerSymbolTableTrait objects must return different instances of inner symbol tables"

    unpopulated_inner_sym_tables = InnerSymbolTableCollection()
    sym_table3 = unpopulated_inner_sym_tables.get_inner_symbol_table(mod1)
    sym_table4 = unpopulated_inner_sym_tables.get_inner_symbol_table(mod2)
    assert (
        sym_table3 is not sym_table4
    ), "InnerSymbolTableTrait still behave as expected when created on the fly"


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
    assert (
        ref.get_module().data == "mod2"
    ), "Name of the referenced module should be returned correctly"


def test_inner_sym_attr():
    """
    Test inner symbol attributes
    """
    invalid_sym_attr = InnerSymAttr()
    assert (
        invalid_sym_attr.get_sym_name() is None
    ), "Invalid InnerSymAttr should return no name"

    sym_attr = InnerSymAttr("sym")
    assert sym_attr.get_sym_name() == StringAttr(
        "sym"
    ), "InnerSymAttr for “ground” type should return name"

    with pytest.raises(VerifyException, match=r"inner symbol cannot have empty name"):
        InnerSymAttr("")

    aggregate_sym_attr = InnerSymAttr(
        [
            InnerSymPropertiesAttr("sym", 0, "public"),
            InnerSymPropertiesAttr("other", 1, "private"),
            InnerSymPropertiesAttr("yet_another", 2, "nested"),
        ]
    )

    assert aggregate_sym_attr.get_sym_name() == StringAttr(
        "sym"
    ), "InnerSymAttr for aggregate types should return name with field ID 0"
