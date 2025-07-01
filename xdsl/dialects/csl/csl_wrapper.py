from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import cast

from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    NoneAttr,
    StringAttr,
)
from xdsl.dialects.csl import csl
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
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
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
    Represents a module parameter that needs to have a type, and may have a value.
    """

    name = "csl_wrapper.param"

    key: StringAttr
    value: IntegerAttr[IntegerType] | NoneAttr
    type: IntegerType

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.key.data)
            if not isinstance(self.value, NoneAttr):
                printer.print_string(" default=")
                printer.print_attribute(self.value)
            else:
                printer.print_string(" : ")
                printer.print_attribute(self.type)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            key = StringAttr(parser.parse_str_literal())
            if parser.parse_optional_keyword("default"):
                parser.parse_punctuation("=")
                val = parser.parse_attribute()
                assert isa(val, IntegerAttr)
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
class ImportOp(IRDLOperation):
    """
    Lightweight wrapper around `csl.import_module` that allows specifying field names directly
    and removes the need for handling structs or setting up struct operands.

    Where existing structs need to be used in the import, they can be passed
    with an empty field name. This will concatenate them all together.

    Named fields and empty fields can be used in the same import
    """

    name = "csl_wrapper.import"

    ops = var_operand_def()
    module = prop_def(StringAttr)
    fields = prop_def(ArrayAttr[StringAttr])
    result = result_def(csl.ImportedModuleType)

    def __init__(
        self, module: str, field_name_mapping: dict[str, Operation | SSAValue]
    ):
        ops: list[Operation | SSAValue] = []
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
            raise VerifyException("Number of fields does not match number of operands")


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

    The layout module has the following args (in order):
      * set_tile_code params:  `x` and `y`
      * general params:        `width` and `height` followed by everything specified in `params`

    The program module has the following args (in order):
      * general params:        `width` and `height` followed by everything specified in `params`
      * params from layout:    everything defined by `layout_yield_op.fields`
    """

    name = "csl_wrapper.module"

    width = prop_def(IntegerAttr[IntegerType])
    height = prop_def(IntegerAttr[IntegerType])
    program_name = opt_prop_def(StringAttr)
    target = prop_def(StringAttr)
    params = prop_def(ArrayAttr[ParamAttribute])

    layout_module = region_def("single_block")
    program_module = region_def("single_block")

    def __init__(
        self,
        width: int | IntegerAttr[IntegerType],
        height: int | IntegerAttr[IntegerType],
        target: csl.Target | StringAttr,
        params: (
            dict[str, IntegerAttr[IntegerType]] | Sequence[ParamAttribute] | None
        ) = None,
    ):
        if not isinstance(width, IntegerAttr):
            width = IntegerAttr(width, i16)
        if not isinstance(height, IntegerAttr):
            height = IntegerAttr(height, i16)
        if not isinstance(target, StringAttr):
            target = StringAttr(target)
        if params is None:
            params = []
        elif isinstance(params, dict):
            params = [
                ParamAttribute(StringAttr(name), val, val.type)
                for name, val in params.items()
            ]
        params_attr = ArrayAttr(params)

        super().__init__(
            properties={
                "width": width,
                "height": height,
                "params": params_attr,
                "target": target,
            },
            regions=[
                Region(
                    Block(
                        arg_types=itertools.chain(
                            [i16] * 4,
                            (param.type for param in params),
                        )
                    )
                ),
                Region(
                    Block(
                        arg_types=itertools.chain(
                            [i16] * 2,
                            (param.type for param in params),
                        )
                    )
                ),
            ],
        )

    def update_program_block_args(
        self,
        yield_args: Iterable[tuple[str, SSAValue]] | None = None,
    ):
        """
        Update `program_module` BlockArguments by adding
        1. yield op fields (pass None to enable automated retrieval, pass empty list to add no yield op fields)
        2. additional exported symbols
        """
        assert (
            len(self.program_module.block.args)
            == len(self.layout_module.block.args) - 2
            # minus two as layout_module has additional x and y args
        ), (
            "program_module block args should only contain args from properties when calling this function"
        )

        if yield_args is None:
            yield_args = self.layout_yield_op.items()

        for name, op in yield_args:
            arg = self.program_module.block.insert_arg(
                op.type, len(self.program_module.block.args)
            )
            arg.name_hint = name

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
            self.layout_module.block.arg_types[4:],
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
        # these may be followed by input-output symbols which we cannot verify, therefore setting `strict=False`
        for got, (name, exp) in zip(
            self.program_module.block.arg_types[2:],
            itertools.chain(
                (
                    (param.key.data, cast(Attribute, param.type))
                    for param in self.params
                ),
                ((key, val.type) for key, val in self.layout_yield_op.items()),
            ),
            strict=False,
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

    def get_program_param(self, name: str) -> BlockArgument:
        """Retrieve program block arg for name that is one of the properties or a param set up by layout yield"""
        # check static params
        if name in ("width", "height"):
            return self.program_module.block.args[("width", "height").index(name)]
        # check module params
        for i, param in enumerate(self.params):
            if param.key.data == name:
                return self.program_module.block.args[2 + i]
        # check yielded params:
        for i, (key, _) in enumerate(self.layout_yield_op.items()):
            if key == name:
                return self.program_module.block.args[2 + len(self.params) + i]
        # not found = value error
        raise ValueError(f"{name} does not refer to a block arg of this program_module")

    def get_param_value(self, name: str) -> IntegerAttr[IntegerType]:
        """Retrieve the value of a named op param."""
        if name == "width":
            return self.width
        elif name == "height":
            return self.height
        res = NoneAttr()
        for param in self.params.data:
            if name == param.key.data:
                res = param.value
        if isinstance(res, NoneAttr):
            raise ValueError(f"Parameter name is unknown or has no value: {name}")
        return res

    @property
    def layout_yield_op(self) -> YieldOp:
        """
        Get the yield op from the layout module. Used in various places.
        """
        return cast(YieldOp, self.layout_module.block.last_op)

    @property
    def exported_symbols(self) -> Sequence[BlockArgument]:
        """
        Get the exported symbols.
        """
        return self.program_module.block.args[
            2 + len(self.params) + len(self.layout_yield_op.fields) :
        ]

    def get_program_import(self, name: str) -> ImportOp:
        """Get top-level import op in the program_module"""
        for op in self.program_module.ops:
            if isinstance(op, ImportOp) and op.module.data == name:
                return op
        raise ValueError(f"Cannot get program_module import of {name}")


@irdl_op_definition
class YieldOp(IRDLOperation):
    """
    Layout module `yield` ops should be lowered to `@set_tile_code` and must specify `fields`.
    Program module `yield` ops have no particular meaning and specify `fields` here is permitted but undefined.
    """

    name = "csl_wrapper.yield"

    values = var_operand_def(Attribute)
    fields = prop_def(ArrayAttr[StringAttr])

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(ModuleOp),
            Pure(),
        )
    )

    def __init__(
        self,
        operands: Sequence[SSAValue | Operation],
        args: Sequence[str] | ArrayAttr[StringAttr],
    ):
        if not isinstance(args, ArrayAttr):
            args = ArrayAttr(StringAttr(arg) for arg in args)
        super().__init__(
            operands=[operands],
            properties={
                "fields": args,
            },
        )

    @staticmethod
    def from_field_name_mapping(field_name_mapping: dict[str, Operation | SSAValue]):
        operands: list[Operation | SSAValue] = []
        attributes: list[StringAttr] = []
        for attr, op in field_name_mapping.items():
            attributes.append(StringAttr(attr))
            operands.append(op)
        return YieldOp(operands, ArrayAttr[StringAttr](attributes))

    def verify_(self) -> None:
        if len(self.fields) != len(self.operands):
            raise VerifyException("Number of fields must match the number of operands")

    def items(self) -> Iterable[tuple[str, SSAValue]]:
        assert self.fields is not None
        return zip((elm.data for elm in self.fields.data), self.operands, strict=True)


CSL_WRAPPER = Dialect(
    "csl_wrapper",
    [
        ImportOp,
        ModuleOp,
        YieldOp,
    ],
    [
        ParamAttribute,
    ],
)
