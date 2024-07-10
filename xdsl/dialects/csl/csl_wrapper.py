from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.dialects.csl import csl
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import Attribute, Block, Dialect, Operation, Region, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    irdl_op_definition,
    opt_attr_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsTerminator, Pure
from xdsl.utils.hints import isa


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
    params = prop_def(DictionaryAttr)

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
        if "x" in self.properties or "y" in self.properties:
            raise ValueError("'x' and 'y' not allowed as property names")

        all_params: dict[str, Attribute] = {"width": self.width, "height": self.height}
        if len(self.params.data) > 0:
            all_params.update(self.params.data)

        expected_layout_args: dict[str, IntegerType] = {
            "x": IntegerType(16),
            "y": IntegerType(16),
        }
        expected_program_args: dict[str, Attribute] = {}
        for name, attr in all_params.items():
            if not isa(attr, IntegerAttr[IntegerType]):
                raise ValueError(f"property {name} must be IntegerAttr[IntegerType]")
            expected_layout_args[name] = attr.type
            expected_program_args[name] = attr.type

        for got, (name, exp) in zip(
            [a.type for a in self.layout_module.block.args],
            expected_layout_args.items(),
            strict=True,
        ):
            if exp != got:
                raise ValueError(
                    f"Layout module block arg types do not match for arg {name} expected: {exp} but got: {got}. Block arg types must correspond to prop types (in order)"
                )

        yield_op = self.layout_module.block.last_op
        if not isinstance(yield_op, YieldOp):
            raise ValueError("layout module must be terminated by csl_wrapper.yield")

        if yield_op.fields is None:
            raise ValueError("layout module yield must specify fields property")
        yield_op.verify_()
        for name, op in zip(yield_op.fields, yield_op.arguments):
            expected_program_args[name.data] = op.type

        for got, (name, exp) in zip(
            [a.type for a in self.program_module.block.args],
            expected_program_args.items(),
            strict=True,
        ):
            if exp != got:
                raise ValueError(
                    f"Program module block arg types do not match for arg {name} expected: {exp} but got: {got}. Block arg types must correspond to prop types and layout yield result types (in order)"
                )

    @classmethod
    def parse(cls, parser: Parser):
        args = parser.parse_op_args_list()
        operands = parser.resolve_operands(args, [], parser.pos)

        props = parser.parse_optional_properties_dict()
        props_l = list(props.items())
        assert len(props_l) >= 2
        name, width = props_l.pop(0)
        assert name == "width"
        name, height = props_l.pop(0)
        assert name == "height"
        params = dict[str, Attribute]()
        for name, value in props_l:
            params[name] = value

        parser.parse_punctuation("(")
        layout_module = parser.parse_region()
        parser.parse_punctuation(",")
        program_module = parser.parse_region()
        parser.parse_punctuation(")")

        return cls(
            operands=operands,
            result_types=[],
            regions=[layout_module, program_module],
            properties={
                "width": width,
                "height": height,
                "params": DictionaryAttr(data=params),
            },
            attributes={},
        )

    def print(self, printer: Printer):
        printer.print("() ")

        params: dict[str, Attribute] = {"width": self.width, "height": self.height}
        params.update(self.params.data)

        printer.print("<")
        printer.print_attr_dict(params)
        printer.print("> (")
        printer.print_region(self.layout_module, print_entry_block_args=True)
        printer.print(", ")
        printer.print_region(self.program_module, print_entry_block_args=True)
        printer.print(")")

    def get_layout_arg(self, name: str):
        """Retrieve layout block arg for name that is x, y, or one of the properties"""
        available_arg_names = ["x", "y", "width", "height"] + list(
            self.params.data.keys()
        )
        assert (
            name in available_arg_names
        ), f"{name} does not refer to a block arg of this layout_module"
        idx = available_arg_names.index(name)
        return self.layout_module.block.args[idx]

    def get_program_arg(self, name: str):
        """Retrieve program block arg for name that is one of the properties or a param set up by layout yield"""
        layout_yield = self.layout_module.block.last_op
        assert isinstance(layout_yield, YieldOp)
        yield_args = (
            []
            if layout_yield.fields is None
            else [y.data for y in layout_yield.fields.data]
        )
        available_arg_names = (
            ["width", "height"] + list(self.params.data.keys()) + yield_args
        )
        assert (
            name in available_arg_names
        ), f"{name} does not refer to a block arg of this program_module"
        idx = available_arg_names.index(name)
        return self.program_module.block.args[idx]


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


CSL_WRAPPER = Dialect(
    "csl_wrapper",
    [
        ImportModuleOp,
        ModuleOp,
        YieldOp,
    ],
    [],
)
