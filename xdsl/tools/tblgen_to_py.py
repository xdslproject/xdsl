import argparse
import json
import subprocess
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from keyword import iskeyword
from sys import stdin
from typing import Any


@dataclass
class TblgenRecord:
    js: Any

    def __getitem__(self, i: str) -> Any:
        return self.js[i]

    @property
    def name(self) -> str:
        return self["!name"]

    @property
    def summary(self) -> str:
        return self["summary"]

    @property
    def superclasses(self) -> set[str]:
        return set(self["!superclasses"])

    @property
    def dialect(self) -> str:
        return self["dialect"]["def"]


class TblgenOp(TblgenRecord):
    @property
    def dialect(self) -> str:
        return self["opDialect"]["def"]

    @property
    def op_name(self) -> str:
        return self["opName"]

    @property
    def assembly_format(self) -> str | None:
        if "assemblyFormat" in self.js:
            assembly = self["assemblyFormat"]
            if isinstance(assembly, str):
                return assembly
        return None

    def _dag_to_py(self, dag: str) -> tuple[tuple[str, str], ...]:
        args = self[dag]["args"]
        return tuple((a["def"], n) for a, n in args)

    @property
    def arguments(self) -> tuple[tuple[str, str], ...]:
        return self._dag_to_py("arguments")

    @property
    def results(self) -> tuple[tuple[str, str], ...]:
        return self._dag_to_py("results")

    @property
    def regions(self) -> tuple[tuple[str, str], ...]:
        return self._dag_to_py("regions")

    @property
    def successors(self) -> tuple[tuple[str, str], ...]:
        return self._dag_to_py("successors")


class TblgenType(TblgenRecord):
    @property
    def type_name(self) -> str:
        return self["typeName"]


class TblgenAttr(TblgenRecord):
    @property
    def attr_name(self) -> str:
        return self["attrName"]


@dataclass
class TblgenLoader:
    """
    Class for converting a json generated by `llvm-tblgen --dump-json` to python
    code which builds an xDSL dialect. Must be initialised with json represented
    as a python object. The generated code is stored as strings in the
    `attributes` and `operations` dictionaries.
    """

    js: Any

    attributes: dict[str, str] = field(default_factory=dict)
    operations: dict[str, str] = field(default_factory=dict)
    used_records: set[str] = field(default_factory=set)
    anon_counter: int = field(default_factory=int)

    def _get_op(self, name: str) -> TblgenOp:
        self.used_records.add(name)
        return TblgenOp(self.js[name])

    def _get_type(self, name: str) -> TblgenType:
        self.used_records.add(name)
        return TblgenType(self.js[name])

    def _get_attr(self, name: str) -> TblgenAttr:
        self.used_records.add(name)
        return TblgenAttr(self.js[name])

    def _get_record(self, name: str) -> TblgenRecord:
        self.used_records.add(name)
        return TblgenRecord(self.js[name])

    def generate_dialect(self, tblgen_dialect: str):
        """
        Generate a dialect from the json object, generating all its contained
        operations, types, and attributes and generating python code which
        is stored in this class' fields.
        """
        self.used_records.add(tblgen_dialect)
        dialect_name = self.js[tblgen_dialect]["name"]

        # Get types
        all_types = self.js["!instanceof"]["TypeDef"]
        for t in all_types:
            ty = self._get_type(t)
            if ty.dialect == tblgen_dialect:
                self.generate_type(ty)

        # Get attributes
        all_attrs = self.js["!instanceof"]["AttrDef"]
        for a in all_attrs:
            attr = self._get_attr(a)
            if attr.dialect == tblgen_dialect:
                self.generate_attr(attr)

        # Get ops
        all_ops = self.js["!instanceof"]["Op"]
        for o in all_ops:
            op = self._get_op(o)
            if op.dialect == tblgen_dialect:
                self.generate_op(op, dialect_name)

    def generate_type(self, tblgen_type: TblgenType):
        """
        Generate a type from the json object, storing python code for it in
        `self.attributes`.
        """

        string = textwrap.dedent(f'''
        @irdl_attr_definition
        class {tblgen_type.name}(ParametrizedAttribute, TypeAttribute):
           """{tblgen_type.summary}"""
           name = "{tblgen_type.type_name}"
        ''')

        self.attributes[tblgen_type.name] = string

    def generate_attr(self, tblgen_attr: TblgenAttr):
        """
        Generate an attribute from the json object, storing python code for it in
        `self.attributes`.
        """

        string = textwrap.dedent(f'''
        @irdl_attr_definition
        class {tblgen_attr.name}(ParametrizedAttribute):
           """{tblgen_attr.summary}"""
           name = "{tblgen_attr.attr_name}"
        ''')

        self.attributes[tblgen_attr.name] = string

    class _ArgType(Enum):
        SINGLE = 0
        VARIADIC = 1
        OPTIONAL = 2
        PROP = 3
        OPTIONAL_PROP = 4

    def _resolve_type_constraint(self, rec: TblgenRecord | str) -> str:
        if isinstance(rec, str):
            rec = self._get_record(rec)
        if rec.name in self.attributes:
            return f"BaseAttr({rec.name})"

        # match specific types
        match rec.name:
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
                pass
        if "AnyTypeOf" in rec.superclasses:
            return textwrap.dedent(f"""
            AnyOf(
                (
                    {",".join(self._resolve_type_constraint(x["def"]) for x in rec["allowedTypes"])}
                )
            )
            """)

        if "AllOfType" in rec.superclasses:
            return textwrap.dedent(f"""
            AllOf(
                (
                    {",".join(self._resolve_type_constraint(x["def"]) for x in rec["allowedTypes"])}
                )
            )
            """)

        if "AnyI" in rec.superclasses:
            return textwrap.dedent(f"""
            ParamAttrConstraint(
                IntegerType,
                (EqAttrConstraint(IntAttr({rec["bitwidth"]})), AnyAttr()),
            )
            """)

        if "I" in rec.superclasses:
            return f"EqAttrConstraint(IntegerType({rec['bitwidth']}))"
        if "SI" in rec.superclasses:
            return (
                f"EqAttrConstraint(IntegerType({rec['bitwidth']}, Signedness.SIGNED))"
            )
        if "UI" in rec.superclasses:
            return (
                f"EqAttrConstraint(IntegerType({rec['bitwidth']}, Signedness.UNSIGNED))"
            )
        if "Complex" in rec.superclasses:
            return textwrap.dedent(f"""
            ParamAttrConstraint(
                ComplexType,
                ({self._resolve_type_constraint(rec["elementType"]["def"])},),
            )
            """)

        return "AnyAttr()"

    def _resolve_prop_constraint(self, rec: TblgenRecord | str) -> str:
        if isinstance(rec, str):
            rec = self._get_record(rec)

        if rec.name in self.attributes:
            return f"BaseAttr({rec.name})"

        match rec.name:
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
                pass

        if "AnyAttrOf" in rec.superclasses:
            return textwrap.dedent(f"""
            AnyOf(
                {",".join(self._resolve_prop_constraint(x["def"]) for x in rec["allowedAttributes"])}
                )
            )
            """)

        if (
            "AnyIntegerAttrBase" in rec.superclasses
            or "SignlessIntegerAttrBase" in rec.superclasses
            or "SignedIntegerAttrBase" in rec.superclasses
            or "UnsignedIntegerAttrBase" in rec.superclasses
        ):
            return textwrap.dedent(f"""
            ParamAttrConstraint(
                IntegerAttr,
                (
                    AnyAttr(),
                    {self._resolve_type_constraint(rec["valueType"]["def"])},
                ),
            )
            """)

        if "FloatAttrBase" in rec.superclasses:
            return textwrap.dedent(f"""
            ParamAttrConstraint(
                FloatAttr,
                (
                    AnyAttr(),
                    {self._resolve_type_constraint(rec["valueType"]["def"])},
                ),
            )
            """)

        return "AnyAttr()"

    def _resolve_name(self, name: Any) -> str:
        if isinstance(name, str):
            if iskeyword(name):
                return f"{name}_"
            return name

        self.anon_counter += 1
        return f"v{self.anon_counter}"

    def _resolve_constraint(self, rec: TblgenRecord | str) -> tuple[_ArgType, str]:
        if isinstance(rec, str):
            rec = self._get_record(rec)

        superclasses = rec.superclasses
        if "Variadic" in superclasses:
            return (
                self._ArgType.VARIADIC,
                self._resolve_type_constraint(rec["baseType"]["def"]),
            )
        elif "Optional" in superclasses:
            return (
                self._ArgType.OPTIONAL,
                self._resolve_type_constraint(rec["baseType"]["def"]),
            )
        elif "Type" in superclasses:
            return (self._ArgType.SINGLE, self._resolve_type_constraint(rec))
        elif "OptionalAttr" in superclasses:
            return (
                self._ArgType.OPTIONAL_PROP,
                self._resolve_prop_constraint(rec["baseAttr"]),
            )
        else:
            return (self._ArgType.PROP, self._resolve_prop_constraint(rec))

    def generate_op(self, tblgen_op: TblgenOp, dialect_name: str):
        """
        Generate an operation from the json object, storing python code for it in
        `self.operations`.
        """

        fields = {"name": f'"{dialect_name}.{tblgen_op.op_name}"'}

        assembly = tblgen_op.assembly_format
        if assembly is not None and "custom" not in assembly:
            fields["assembly_format"] = assembly

        for [arg, orig_name] in tblgen_op.arguments:
            name = self._resolve_name(orig_name)
            (variadicity, constraint) = self._resolve_constraint(arg)
            match variadicity:
                case self._ArgType.SINGLE:
                    fields[name] = f"operand_def({constraint})"
                case self._ArgType.OPTIONAL:
                    fields[name] = f"opt_operand_def({constraint})"
                case self._ArgType.VARIADIC:
                    fields[name] = f"var_operand_def({constraint})"
                case self._ArgType.PROP:
                    name_str = (
                        f', prop_name = "{orig_name}"' if iskeyword(orig_name) else ""
                    )
                    fields[name] = f"prop_def({constraint}{name_str})"
                case self._ArgType.OPTIONAL_PROP:
                    name_str = (
                        f', prop_name = "{orig_name}"' if iskeyword(orig_name) else ""
                    )
                    fields[name] = f"opt_prop_def({constraint}{name_str})"

        for [res, name] in tblgen_op.results:
            name = self._resolve_name(name)
            (variadicity, constraint) = self._resolve_constraint(res)
            match variadicity:
                case self._ArgType.SINGLE:
                    fields[name] = f"result_def({constraint})"
                case self._ArgType.OPTIONAL:
                    fields[name] = f"opt_result_def({constraint})"
                case self._ArgType.VARIADIC:
                    fields[name] = f"var_result_def({constraint})"
                case _:
                    continue

        for [region, name] in tblgen_op.regions:
            name = self._resolve_name(name)
            region = self._get_record(region)
            variadic = "VariadicRegion" in region.superclasses
            single_block = (
                "SizedRegion" in region.superclasses
                and region.summary == "region with 1 blocks"
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

        for [succ, name] in tblgen_op.successors:
            name = self._resolve_name(name)
            succ = self._get_record(succ)
            if "VariadicRegion" in succ.superclasses:
                fields[name] = "var_successor_def()"
            else:
                fields[name] = "successor_def()"

        field_string = textwrap.indent(
            "\n\n".join(f"{x} = {d}" for x, d in fields.items()), "    "
        )
        string = f'''
@irdl_op_definition
class {tblgen_op.name}(IRDLOperation):
    {"" if tblgen_op.summary == "" else f'"""{tblgen_op.summary}"""'}

{field_string}
'''

        self.operations[tblgen_op.name] = string


def main():
    # Parse CLI arguments
    arg_parser = argparse.ArgumentParser(
        description="Convert tblgen json to a Python definition of a xDSL dialect."
    )
    arg_parser.add_argument(
        "-o", "--output-file", required=False, type=str, help="path to output file"
    )
    arg_parser.add_argument(
        "-i", "--input_file", required=False, type=str, help="path to input file"
    )
    arg_parser.add_argument(
        "-c",
        "--cull",
        action="store_true",
        help="Output a culled json with only necessary fields",
    )
    args = arg_parser.parse_args()

    if args.input_file is None:
        in_file = stdin
    else:
        in_file = open(args.input_file)

    with in_file as file:
        js = json.load(file)
    loader = TblgenLoader(js)
    dialects = js["!instanceof"]["Dialect"]
    [dialect] = dialects
    loader.generate_dialect(dialect)

    if args.cull:
        js = loader.js
        required_fields = {
            "!name",
            "!superclasses",
            "assemblyFormat",
            "summary",
            "dialect",
            "opDialect",
            "typeName",
            "attrName",
            "opName",
            "arguments",
            "results",
            "regions",
            "successors",
            "allowedTypes",
            "bitwidth",
            "elementType",
            "valueType",
            "baseType",
            "baseAttr",
            "def",
            "name",
        }

        def cull_field(js_in: dict[str, Any]) -> dict[str, Any]:
            return {key: js_in[key] for key in js_in if key in required_fields}

        culled: dict[str, Any] = {
            key: cull_field(js[key]) for key in loader.used_records
        }
        culled["!instanceof"] = {
            key: js["!instanceof"][key]
            for key in ("TypeDef", "AttrDef", "Op", "Dialect")
        }

        if args.output_file is not None:
            with open(args.output_file, "w") as out_file:
                print(json.dumps(culled), file=out_file)
        else:
            print(json.dumps(culled))
    else:
        with StringIO() as out_str:
            print(
                textwrap.dedent(f"""\
                \"""
                This file is automatically generated by xDSL and not meant to be modified.

                It was generated from {args.input_file}
                \"""

                # ruff: noqa: F403, F405

                from xdsl.dialects.builtin import *
                from xdsl.ir import *
                from xdsl.irdl import *
                """),
                file=out_str,
            )

            for attr in loader.attributes.values():
                print(attr, file=out_str)

            for op in loader.operations.values():
                print(op, file=out_str)

            content = out_str.getvalue()

        # Format output
        output = subprocess.run(
            [
                "ruff",
                "format",
                "--stdin-filename",
                f"{dialect}.py",
            ],
            input=content,
            capture_output=True,
            text=True,
        )

        if args.output_file is not None:
            with open(args.output_file, "w") as out_file:
                print(output.stdout, file=out_file)
        else:
            print(output.stdout)
