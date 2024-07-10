from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import cast

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
)
from xdsl.dialects.csl import ParameterDef, csl
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    irdl_attr_definition,
    irdl_op_definition,
    opt_attr_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

i16 = builtin.IntegerType(16)


@irdl_attr_definition
class ParamAttribute(ParametrizedAttribute):
    """
    TODO: better name here

    Represents a module parameter that needs to have a type, and may have a value.
    """

    name = "csl_wrapper.param"

    key: ParameterDef[StringAttr]
    value: ParameterDef[IntegerAttr[IntegerType] | NoneAttr]
    type: ParameterDef[IntegerType]

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.key.data)
            if not isinstance(self.value, NoneAttr):
                printer.print(" default=")
                printer.print_attribute(self.value)
            else:
                printer.print(" : ")
                printer.print_attribute(self.type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            key = StringAttr(parser.parse_str_literal())
            if parser.parse_optional_keyword("default"):
                parser.parse_punctuation("=")
                val = parser.parse_attribute()
                assert isa(val, AnyIntegerAttr)
                assert isinstance(val.type, IntegerType)
                type = val.type
            else:
                parser.parse_punctuation(":")
                val = NoneAttr()
                type = parser.parse_type()
        return key, val, type

    def verify(self) -> None:
        super().verify()
        if not isinstance(self.value, NoneAttr):
            if self.value.type != self.type:
                raise VerifyException(
                    f"Value expected to be of type {self.type}, found {self.value.type}"
                )


@irdl_op_definition
class ImportModuleOp(IRDLOperation):
    """
    Lightweight wrapper around `csl.import_module` that allows specifying field names directly
    and removes the need for handling structs or setting up struct operands
    """

    name = "csl_wrapper.import_module"

    ops = var_operand_def()
    module = prop_def(StringAttr)
    fields = prop_def(ArrayAttr[StringAttr])
    result = result_def(csl.ImportedModuleType)

    def __init__(self, module: str, field_name_mapping: dict[str, Operand | SSAValue]):
        ops: list[Operand | SSAValue] = []
        fields: list[StringAttr] = []
        for field, op in field_name_mapping.items():
            ops.append(op)
            fields.append(StringAttr(field))

        super().__init__(
            operands=[ops],
            properties={
                "module": StringAttr(module),
                "fields": ArrayAttr(fields),
            },
            result_types=[csl.ImportedModuleType()],
        )

    def verify_(self) -> None:
        if len(self.fields) != len(self.ops):
            raise ValueError("Number of fields does not match number of operands")


@irdl_op_definition
class ModuleOp(IRDLOperation):
    """
    Wrapper class that manages initialisation of layout and program module.

    Specified properties will be lowered to module params (`csl.param`). As such, all properties
    are passed as BlockArgs to the `layout_module` and `program_module` region. The order in which
    properties are specified is important and must match the order of the block args.

    Additionally, any value yielded by the `layout_module` is passed as a block arg (lowered to `csl.param`)
    to the program module. Order is important here as well. The layout module's `yield` op should be lowered to
    `@set_tile_code`, while the program module's `yield` op should be discarded.

    The layout module has two additional block args `x` and `y` as part of the `@set_tile_code` loop nest.
    Operations using these args need to be lowered to the correct place in the loop nest.
    """

    name = "csl_wrapper.module"

    width = prop_def(IntegerAttr)
    height = prop_def(IntegerAttr)
    params: ArrayAttr[ParamAttribute] = prop_def(ArrayAttr[ParamAttribute])

    layout_module = region_def("single_block")
    program_module = region_def("single_block")

    @staticmethod
    def from_properties(properties: dict[str, Attribute]):
        layout_module = Region(Block())
        program_module = Region(Block())
        x = layout_module.block.insert_arg(IntegerType(16), 0)
        y = layout_module.block.insert_arg(IntegerType(16), 1)
        x.name_hint = "x"
        y.name_hint = "y"

        for name, value in properties.items():
            if not isa(value, IntegerAttr[IntegerType]):
                raise ValueError("Can only create module from IntegerAttr properties")
            l_arg = layout_module.block.insert_arg(
                value.type, len(layout_module.block.args)
            )
            l_arg.name_hint = name
            p_arg = program_module.block.insert_arg(
                value.type, len(program_module.block.args)
            )
            p_arg.name_hint = name

        props = {
            "width": properties.pop("width"),
            "height": properties.pop("height"),
            "params": DictionaryAttr(properties),
        }

        return ModuleOp(properties=props, regions=[layout_module, program_module])

    def update_program_block_args_from_layout(self):
        """Update `program_module` BlockArguments by adding yield op fields"""
        assert (
            len(self.program_module.block.args)
            == len(self.layout_module.block.args) - 2
            # minus two as layout_module has additional x and y args
        ), "program_module block args should only contain args from properties when calling this function"

        yield_op = self.layout_module.block.last_op
        if not isinstance(yield_op, YieldOp):
            raise ValueError("layout module must be terminated by csl_wrapper.yield")

        if yield_op.fields is None:
            raise ValueError("layout module yield must specify fields property")
        yield_op.verify_()
        for name, op in zip(yield_op.fields, yield_op.arguments):
            arg = self.program_module.block.insert_arg(
                op.type, len(self.program_module.block.args)
            )
            arg.name_hint = name.data

    def verify_(self):
        # verify that names are unique
        names: set[str] = {"x", "y", "width", "height"}
        for param in self.params.data:
            if param.key.data in names:
                raise VerifyException(f"Duplicate name in parameters: {param.key.data}")
            names.add(param.key.data)

        # verify that x, y, width, height are i16
        if not all(arg.type == i16 for arg in self.layout_module.block.args[:4]):
            raise VerifyException(
                "The first four arguments of the layout block (x, y, width, height) must be of type i16"
            )

        # verify that block args are of the right type for the provided params
        for arg, param in zip(
            [a.type for a in self.layout_module.block.args[4:]],
            self.params,
            strict=True,
        ):
            if arg != param.type:
                raise ValueError(
                    f"Layout module block arg types do not match for arg {param.key} expected: {param.type} but got: "
                    f"{arg}. Block arg types must correspond to prop types (in order)"
                )

        # verify that the first two program block args (width, height) are correctly typed
        if not all(arg.type == i16 for arg in self.program_module.block.args[:2]):
            raise VerifyException(
                "The first two arguments of the program block (width, height) must be of type i16"
            )

        # verify that params and yielded arguments are typed correctly
        for got, (name, exp) in zip(
            [a.type for a in self.program_module.block.args[2:]],
            itertools.chain(
                (
                    (param.key.data, cast(Attribute, param.type))
                    for param in self.params
                ),
                ((key, val.type) for key, val in self.layout_yield_op.items()),
            ),
            strict=True,
        ):
            if exp != got:
                raise VerifyException(
                    f"Program module block arg types do not match for arg {name} expected: {exp} but got: {got}. "
                    f"Block arg types must correspond to prop types and layout yield result types (in order)"
                )

    def get_layout_param(self, name: str) -> BlockArgument:
        """
        Retrieve layout block arg for name that is x, y, or one of the properties
        """
        # check static params:
        if name in ("x", "y", "width", "height"):
            return self.layout_module.block.args[
                ("x", "y", "width", "height").index(name)
            ]
        # check module params
        for i, param in enumerate(self.params):
            if param.key.data == name:
                return self.layout_module.block.args[4 + i]
        # not found = value error
        raise ValueError(f"{name} does not refer to a block arg of this layout_module")

    def get_program_param_arg(self, name: str) -> BlockArgument:
        """Retrieve program block arg for name that is one of the properties or a param set up by layout yield"""
        # check static params
        if name in ("width", "height"):
            return self.layout_module.block.args[("width", "height").index(name)]
        # check module params
        for i, param in enumerate(self.params):
            if param.key.data == name:
                return self.layout_module.block.args[2 + i]
        # check yielded params:
        for i, (key, _) in enumerate(self.layout_yield_op.items()):
            if key == name:
                return self.layout_module.block.args[2 + len(self.params) + i]
        # not found = value error
        raise ValueError(f"{name} does not refer to a block arg of this program_module")

    @property
    def layout_yield_op(self) -> YieldOp:
        """
        Get the yield op from the layout module. Used in various places.
        """
        return cast(YieldOp, self.layout_module.block.last_op)


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    """
    Layout module `yield` ops should be lowered to `@set_tile_code` and must specify `fields`.
    Program module `yield` ops have no particular meaning and specify `fields` here is permitted but undefined.
    """

    name = "csl_wrapper.yield"

    fields = opt_attr_def(ArrayAttr[StringAttr])

    traits = traits_def(
        lambda: frozenset([IsTerminator(), HasParent(ModuleOp), Pure()])
    )

    def __init__(self, *operands: SSAValue | Operation):
        super().__init__(*operands)

    @staticmethod
    def from_field_name_mapping(field_name_mapping: dict[str, Operand | SSAValue]):
        operands: list[Operand | SSAValue] = []
        attributes: list[StringAttr] = []
        for attr, op in field_name_mapping.items():
            attributes.append(StringAttr(attr))
            operands.append(op)
        result = YieldOp(*operands)
        result.attributes.update({"fields": ArrayAttr[StringAttr](attributes)})
        return result

    def verify_(self) -> None:
        if self.fields is not None and len(self.fields) != len(self.operands):
            raise ValueError("Number of fields must match the number of operands")

    def items(self) -> Iterable[tuple[str, SSAValue]]:
        assert self.fields is not None
        return zip((elm.data for elm in self.fields.data), self.operands, strict=True)


CSL_WRAPPER = Dialect(
    "csl_wrapper",
    [
        ImportModuleOp,
        ModuleOp,
        YieldOp,
    ],
    [
        ParamAttribute,
    ],
)
