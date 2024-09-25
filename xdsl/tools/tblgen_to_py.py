import argparse
import json
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from keyword import iskeyword
from typing import Any


@dataclass
class TblgenLoader:
    js: Any

    attributes: dict[str, str] = field(default_factory=dict)
    operations: dict[str, str] = field(default_factory=dict)
    anon_counter: int = field(default_factory=int)

    def load_dialect(self, dialect_name: str):
        # Get types
        all_types = self.js["!instanceof"]["TypeDef"]
        for t in all_types:
            if self.js[t]["dialect"]["def"] == dialect_name:
                self.load_type(t)

        # Get attributes
        all_attrs = self.js["!instanceof"]["AttrDef"]
        for a in all_attrs:
            if self.js[a]["dialect"]["def"] == dialect_name:
                self.load_attr(a)

        # Get ops
        all_ops = self.js["!instanceof"]["Op"]
        for o in all_ops:
            if self.js[o]["opDialect"]["def"] == dialect_name:
                self.load_op(o)

    def load_type(self, type_def: str):
        tblgen_type = self.js[type_def]

        string = textwrap.dedent(f'''
        @irdl_attr_definition
        class {tblgen_type["!name"]}(ParametrizedAttribute, TypeAttribute):
           """{tblgen_type["summary"]}"""
           name = "{tblgen_type["typeName"]}"
        ''')

        self.attributes[tblgen_type["!name"]] = string

    def load_attr(self, attr_def: str):
        tblgen_attr = self.js[attr_def]

        string = textwrap.dedent(f'''
        @irdl_attr_definition
        class {tblgen_attr["!name"]}(ParametrizedAttribute):
           """{tblgen_attr["summary"]}"""
           name = "{tblgen_attr["attrName"]}"
        ''')

        self.attributes[tblgen_attr["!name"]] = string

    class ArgType(Enum):
        SINGLE = 0
        VARIADIC = 1
        OPTIONAL = 2
        PROP = 3
        OPTIONAL_PROP = 4

    def resolve_type_constraint(self, cls_name: str) -> str:
        if cls_name in self.attributes:
            return f"BaseAttr({cls_name})"

        # match specific types
        match cls_name:
            case "NoneType":
                return "EqAttrConstraint(NoneType())"
            case "AnyInteger":
                return "BaseAttr(IntegerType)"
            case "AnySignlessInteger":
                return textwrap.dedent("""
                ParamAttrConstraint(
                    IntegerType,
                    (AnyAttr(), EqAttrConstraint(SignednessAttr(Signedness.SIGNLESS))),
                )""")
            case "AnySignedInteger":
                return textwrap.dedent("""
                ParamAttrConstraint(
                    IntegerType,
                    (AnyAttr(), EqAttrConstraint(SignednessAttr(Signedness.SIGNED))),
                )
                """)
            case "AnyUnsignedInteger":
                return textwrap.dedent("""
                ParamAttrConstraint(
                    IntegerType,
                    (AnyAttr(), EqAttrConstraint(SignednessAttr(Signedness.UNSIGNED))),
                )
                """)
            case "Index":
                return "EqAttrConstraint(IndexType())"
            case "F16":
                return "EqAttrConstraint(Float16Type())"
            case "F32":
                return "EqAttrConstraint(Float32Type())"
            case "F64":
                return "EqAttrConstraint(Float64Type())"
            case "F80":
                return "EqAttrConstraint(Float80Type())"
            case "F128":
                return "EqAttrConstraint(Float128Type())"
            case "BF16":
                return "EqAttrConstraint(BFloat16Type())"
            case "AnyFloat":
                return "AnyFloatConstr"
            case "AnyComplex":
                return "BaseAttr(ComplexType)"
            case _:
                rec = self.js[cls_name]
                if "AnyTypeOf" in rec["!superclasses"]:
                    return textwrap.dedent(f"""
                    AnyOf(
                        (
                            {",".join(self.resolve_type_constraint(x["def"]) for x in rec["allowedTypes"])}
                        )
                    )
                    """)

                if "AllOfType" in rec["!superclasses"]:
                    return textwrap.dedent(f"""
                    AllOf(
                        (
                            {",".join(self.resolve_type_constraint(x["def"]) for x in rec["allowedTypes"])}
                        )
                    )
                    """)

                if "AnyI" in rec["!superclasses"]:
                    return textwrap.dedent(f"""
                    ParamAttrConstraint(
                        IntegerType,
                        (EqAttrConstraint(IntAttr({rec["bitwidth"]})), AnyAttr()),
                    )
                    """)

                if "I" in rec["!superclasses"]:
                    return f"EqAttrConstraint(IntegerType({rec['bitwidth']}))"
                if "SI" in rec["!superclasses"]:
                    return f"EqAttrConstraint(IntegerType({rec['bitwidth']}, Signedness.SIGNED))"
                if "UI" in rec["!superclasses"]:
                    return f"EqAttrConstraint(IntegerType({rec['bitwidth']}, Signedness.UNSIGNED))"
                if "Complex" in rec["!superclasses"]:
                    return textwrap.dedent(f"""
                    ParamAttrConstraint(
                        ComplexType,
                        ({self.resolve_type_constraint(rec["elementType"]["def"])},),
                    )
                    """)

                return "AnyAttr()"

    def resolve_prop_constraint(self, cls_name: str) -> str:
        if cls_name in self.attributes:
            return f"BaseAttr({cls_name})"

        match cls_name:
            case "BoolAttr":
                return "BaseAttr(BoolAttr)"
            case "IndexAttr":
                return textwrap.dedent("""
                ParamAttrConstraint(
                    IntegerAttr, (AnyAttr(), EqAttrConstraint(IndexType()))
                )
                """)

            case "APIntAttr":
                return textwrap.dedent("""
                ParamAttrConstraint(
                    IntegerAttr, (AnyAttr(), AnyAttr())
                )
                """)  # TODO can't represent APInt properly

            case "StrAttr":
                return "BaseAttr(StringAttr)"
            case "SymbolNameAttr":
                return "BaseAttr(SymbolNameAttr)"
            case "UnitAttr":
                return "EqAttrConstraint(UnitAttr())"
            case _:
                rec = self.js[cls_name]
                if "AnyAttrOf" in rec["!superclasses"]:
                    return textwrap.dedent(f"""
                    AnyOf(
                        {",".join(self.resolve_prop_constraint(x["def"]) for x in rec["allowedAttributes"])}
                        )
                    )
                    """)

                if (
                    "AnyIntegerAttrBase" in rec["!superclasses"]
                    or "SignlessIntegerAttrBase" in rec["!superclasses"]
                    or "SignedIntegerAttrBase" in rec["!superclasses"]
                    or "UnsignedIntegerAttrBase" in rec["!superclasses"]
                ):
                    return textwrap.dedent(f"""
                    ParamAttrConstraint(
                        IntegerAttr,
                        (
                            AnyAttr(),
                            {self.resolve_type_constraint(rec["valueType"]["def"])},
                        ),
                    )
                    """)

                if "FloatAttrBase" in rec["!superclasses"]:
                    return textwrap.dedent(f"""
                    ParamAttrConstraint(
                        FloatAttr,
                        (
                            AnyAttr(),
                            {self.resolve_type_constraint(rec["valueType"]["def"])},
                        ),
                    )
                    """)

        return "AnyAttr()"

    def resolve_name(self, name: Any) -> str:
        if isinstance(name, str):
            if iskeyword(name):
                return f"{name}_"
            return name

        self.anon_counter += 1
        return f"v{self.anon_counter}"

    def resolve_constraint(self, cls_name: str) -> tuple[ArgType, str]:
        rec = self.js[cls_name]
        if "Variadic" in rec["!superclasses"]:
            return (
                self.ArgType.VARIADIC,
                self.resolve_type_constraint(rec["baseType"]["def"]),
            )
        elif "Optional" in rec["!superclasses"]:
            return (
                self.ArgType.OPTIONAL,
                self.resolve_type_constraint(rec["baseType"]["def"]),
            )
        elif "Type" in rec["!superclasses"]:
            return (self.ArgType.SINGLE, self.resolve_type_constraint(cls_name))
        elif "OptionalAttr" in rec["!superclasses"]:
            return (
                self.ArgType.OPTIONAL_PROP,
                self.resolve_prop_constraint(rec["baseAttr"]),
            )
        else:
            return (self.ArgType.PROP, self.resolve_prop_constraint(cls_name))

    def load_op(self, op_def: str):
        tblgen_op = self.js[op_def]

        fields = {"name": f'"{tblgen_op["opName"]}"'}

        if "assemblyFormat" in tblgen_op:
            assembly = tblgen_op["assemblyFormat"]
            if isinstance(assembly, str) and not tblgen_op["assemblyFormat"].contains(
                "custom"
            ):
                fields["assembly_format"] = tblgen_op["assemblyFormat"]

        for [arg, name] in tblgen_op["arguments"]["args"]:
            name = self.resolve_name(name)
            (variadicity, constraint) = self.resolve_constraint(arg["def"])
            match variadicity:
                case self.ArgType.SINGLE:
                    fields[name] = f"operand_def({constraint})"
                case self.ArgType.OPTIONAL:
                    fields[name] = f"opt_operand_def({constraint})"
                case self.ArgType.VARIADIC:
                    fields[name] = f"var_operand_def({constraint})"
                case self.ArgType.PROP:
                    fields[name] = f"prop_def({constraint})"
                case self.ArgType.OPTIONAL_PROP:
                    fields[name] = f"opt_prop_def({constraint})"

        for [res, name] in tblgen_op["results"]["args"]:
            name = self.resolve_name(name)
            (variadicity, constraint) = self.resolve_constraint(res["def"])
            match variadicity:
                case self.ArgType.SINGLE:
                    fields[name] = f"result_def({constraint})"
                case self.ArgType.OPTIONAL:
                    fields[name] = f"opt_result_def({constraint})"
                case self.ArgType.VARIADIC:
                    fields[name] = f"var_result_def({constraint})"
                case _:
                    continue

        for [region, name] in tblgen_op["regions"]["args"]:
            name = self.resolve_name(name)
            rec = self.js[region["def"]]
            variadic = "VariadicRegion" in rec["!superclasses"]
            single_block = (
                "SizedRegion" in rec["!superclasses"]
                and rec["summary"] == "region with 1 blocks"
            )
            match (variadic, single_block):
                case (False, False):
                    fields[name] = "region_def()"
                case (False, True):
                    fields[name] = 'region_def("single_block")'
                case (True, False):
                    fields[name] = "var_region_def()"
                case (True, True):
                    fields[name] = 'var_region_def("single_block")'
                case _:
                    pass  # Make pyright happy

        for [succ, name] in tblgen_op["successors"]["args"]:
            name = self.resolve_name(name)
            rec = self.js[succ["def"]]
            if "VariadicRegion" in rec["!superclasses"]:
                fields[name] = "var_successor_def()"
            else:
                fields[name] = "successor_def()"

        field_string = textwrap.indent(
            "\n\n".join(f"{x} = {d}" for x, d in fields.items()), "    "
        )
        string = f'''
@irdl_op_definition
class {tblgen_op["!name"]}(IRDLOperation):
    """{tblgen_op["summary"]}"""

{field_string}
'''

        self.operations[tblgen_op["!name"]] = string


def main():
    # Parse CLI arguments
    arg_parser = argparse.ArgumentParser(
        description="Convert tblgen json to a Python definition of a xDSL dialect."
    )
    arg_parser.add_argument("-o", "--output-file", type=str, help="path to output file")
    arg_parser.add_argument("input_file", type=str, help="path to input file")
    args = arg_parser.parse_args()

    js = json.load(open(args.input_file))
    loader = TblgenLoader(js)
    dialects = js["!instanceof"]["Dialect"]
    [dialect] = dialects
    loader.load_dialect(dialect)

    with open(args.output_file, "w") as stubfile:
        print(
            textwrap.dedent(f"""\
            \"""
            This file is automatically generated by xDSL and not meant to be modified.

            It was generated from {args.input_file}
            \"""

            from xdsl.irdl import *
            from xdsl.ir import *
            from xdsl.dialects.builtin import *
            """),
            file=stubfile,
        )

        for attr in loader.attributes.values():
            print(attr, file=stubfile)

        for op in loader.operations.values():
            print(op, file=stubfile)
