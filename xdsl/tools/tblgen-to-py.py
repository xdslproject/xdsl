import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xdsl.dialects.builtin import (
    AnyFloatConstr,
    BFloat16Type,
    BoolAttr,
    ComplexType,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    NoneType,
    Signedness,
    SignednessAttr,
    StringAttr,
    SymbolNameAttr,
    UnitAttr,
)
from xdsl.ir import Attribute, Dialect, Operation, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    BaseAttr,
    EqAttrConstraint,
    GenericAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    region_def,
    result_def,
    successor_def,
    var_operand_def,
    var_region_def,
    var_result_def,
    var_successor_def,
)


@dataclass(init=False)
class TblgenLoader:
    js: Any

    known_attributes: dict[str, type[Attribute]] = field(default_factory=dict)

    def __init__(self, js: Any):
        self.js = js

    def load_dialect(self, dialect_name: str) -> Dialect:
        # Get types
        all_types = self.js["!instanceof"]["TypeDef"]
        built_types = list(
            self.load_type(t)
            for t in all_types
            if self.js[t]["dialect"]["def"] == dialect_name
        )

        # Get attributes
        all_attrs = self.js["!instanceof"]["AttrDef"]
        built_attrs = list(
            self.load_attr(a)
            for a in all_attrs
            if self.js[a]["dialect"]["def"] == dialect_name
        )

        # Get ops
        all_ops = self.js["!instanceof"]["Op"]
        built_ops = list(
            self.load_op(o)
            for o in all_ops
            if self.js[o]["opDialect"]["def"] == dialect_name
        )

        dialect = Dialect(dialect_name, built_ops, built_attrs + built_types)

        return dialect

    def load_type(self, type_def: str) -> type[Attribute]:
        tblgen_type = self.js[type_def]

        cls = type(
            tblgen_type["cppClassName"],
            (ParametrizedAttribute, TypeAttribute),
            {"__doc__": tblgen_type["summary"], "name": tblgen_type["typeName"]},
        )

        final = irdl_attr_definition(cls)

        self.known_attributes[tblgen_type["!name"]] = final

        return final

    def load_attr(self, attr_def: str) -> type[Attribute]:
        tblgen_attr = self.js[attr_def]

        cls = type(
            tblgen_attr["cppClassName"],
            (ParametrizedAttribute,),
            {"__doc__": tblgen_attr["summary"], "name": tblgen_attr["attrName"]},
        )

        final = irdl_attr_definition(cls)

        self.known_attributes[tblgen_attr["!name"]] = final

        return final

    class ArgType(Enum):
        SINGLE = 0
        VARIADIC = 1
        OPTIONAL = 2
        PROP = 3
        OPTIONAL_PROP = 4

    def resolve_type_constraint(
        self, cls_name: str
    ) -> GenericAttrConstraint[Attribute]:
        if cls_name in self.known_attributes:
            return BaseAttr(self.known_attributes[cls_name])

        # match specific types
        match cls_name:
            case "NoneType":
                return EqAttrConstraint(NoneType())
            case "AnyInteger":
                return BaseAttr(IntegerType)
            case "AnySignlessInteger":
                return ParamAttrConstraint(
                    IntegerType,
                    (AnyAttr(), EqAttrConstraint(SignednessAttr(Signedness.SIGNLESS))),
                )
            case "AnySignedInteger":
                return ParamAttrConstraint(
                    IntegerType,
                    (AnyAttr(), EqAttrConstraint(SignednessAttr(Signedness.SIGNED))),
                )
            case "AnyUnsignedInteger":
                return ParamAttrConstraint(
                    IntegerType,
                    (AnyAttr(), EqAttrConstraint(SignednessAttr(Signedness.UNSIGNED))),
                )
            case "Index":
                return EqAttrConstraint(IndexType())
            case "F16":
                return EqAttrConstraint(Float16Type())
            case "F32":
                return EqAttrConstraint(Float32Type())
            case "F64":
                return EqAttrConstraint(Float64Type())
            case "F80":
                return EqAttrConstraint(Float80Type())
            case "F128":
                return EqAttrConstraint(Float128Type())
            case "BF16":
                return EqAttrConstraint(BFloat16Type())
            case "AnyFloat":
                return AnyFloatConstr
            case "AnyComplex":
                return BaseAttr(ComplexType)
            case _:
                rec = self.js[cls_name]
                if "AnyTypeOf" in rec["!superclasses"]:
                    return AnyOf(
                        tuple(
                            self.resolve_type_constraint(x["def"])
                            for x in rec["allowedTypes"]
                        )
                    )

                if "AllOfType" in rec["!superclasses"]:
                    return AllOf(
                        tuple(
                            self.resolve_type_constraint(x["def"])
                            for x in rec["allowedTypes"]
                        )
                    )

                if "AnyI" in rec["!superclasses"]:
                    return ParamAttrConstraint(
                        IntegerType,
                        (EqAttrConstraint(IntAttr(rec["bitwidth"])), AnyAttr()),
                    )
                if "I" in rec["!superclasses"]:
                    return EqAttrConstraint(IntegerType(rec["bitwidth"]))
                if "SI" in rec["!superclasses"]:
                    return EqAttrConstraint(
                        IntegerType(rec["bitwidth"], Signedness.SIGNED)
                    )
                if "UI" in rec["!superclasses"]:
                    return EqAttrConstraint(
                        IntegerType(rec["bitwidth"], Signedness.UNSIGNED)
                    )
                if "Complex" in rec["!superclasses"]:
                    return ParamAttrConstraint(
                        ComplexType,
                        (self.resolve_type_constraint(rec["elementType"]["def"]),),
                    )

                return AnyAttr()

    def resolve_prop_constraint(
        self, cls_name: str
    ) -> GenericAttrConstraint[Attribute]:
        if cls_name in self.known_attributes:
            return BaseAttr(self.known_attributes[cls_name])

        match cls_name:
            case "BoolAttr":
                return BaseAttr(BoolAttr)
            case "IndexAttr":
                return ParamAttrConstraint(
                    IntegerAttr, (AnyAttr(), EqAttrConstraint(IndexType()))
                )
            case "APIntAttr":
                return ParamAttrConstraint(
                    IntegerAttr, (AnyAttr(), AnyAttr())
                )  # TODO can't represent APInt properly
            case "StrAttr":
                return BaseAttr(StringAttr)
            case "SymbolNameAttr":
                return BaseAttr(SymbolNameAttr)
            case "UnitAttr":
                return EqAttrConstraint(UnitAttr())
            case _:
                rec = self.js[cls_name]
                if "AnyAttrOf" in rec["!superclasses"]:
                    return AnyOf(
                        tuple(
                            self.resolve_prop_constraint(x["def"])
                            for x in rec["allowedAttributes"]
                        )
                    )

                if (
                    "AnyIntegerAttrBase" in rec["!superclasses"]
                    or "SignlessIntegerAttrBase" in rec["!superclasses"]
                    or "SignedIntegerAttrBase" in rec["!superclasses"]
                    or "UnsignedIntegerAttrBase" in rec["!superclasses"]
                ):
                    return ParamAttrConstraint(
                        IntegerAttr,
                        (
                            AnyAttr(),
                            self.resolve_type_constraint(rec["valueType"]["def"]),
                        ),
                    )

                if "FloatAttrBase" in rec["!superclasses"]:
                    return ParamAttrConstraint(
                        FloatAttr,
                        (
                            AnyAttr(),
                            self.resolve_type_constraint(rec["valueType"]["def"]),
                        ),
                    )

        return AnyAttr()

    def resolve_constraint(
        self, cls_name: str
    ) -> tuple[ArgType, GenericAttrConstraint[Attribute]]:
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

    def load_op(self, op_def: str) -> type[Operation]:
        tblgen_op = self.js[op_def]

        fields = {"__doc__": tblgen_op["summary"], "name": tblgen_op["opName"]}

        if "assemblyFormat" in tblgen_op and not tblgen_op["assemblyFormat"].contains(
            "custom"
        ):
            fields["assembly_format"] = tblgen_op["assemblyFormat"]

        for [arg, name] in tblgen_op["arguments"]["args"]:
            (variadicity, constraint) = self.resolve_constraint(arg["def"])
            match variadicity:
                case self.ArgType.SINGLE:
                    fields[name] = operand_def(constraint)
                case self.ArgType.OPTIONAL:
                    fields[name] = opt_operand_def(constraint)
                case self.ArgType.VARIADIC:
                    fields[name] = var_operand_def(constraint)
                case self.ArgType.PROP:
                    fields[name] = prop_def(constraint)
                case self.ArgType.OPTIONAL_PROP:
                    fields[name] = opt_prop_def(constraint)

        for [res, name] in tblgen_op["results"]["args"]:
            (variadicity, constraint) = self.resolve_constraint(res["def"])
            match variadicity:
                case self.ArgType.SINGLE:
                    fields[name] = result_def(constraint)
                case self.ArgType.OPTIONAL:
                    fields[name] = opt_result_def(constraint)
                case self.ArgType.VARIADIC:
                    fields[name] = var_result_def(constraint)
                case _:
                    continue

        for [region, name] in tblgen_op["regions"]["args"]:
            rec = self.js[region["def"]]
            variadic = "VariadicRegion" in rec["!superclasses"]
            single_block = (
                "SizedRegion" in rec["!superclasses"]
                and rec["summary"] == "region with 1 blocks"
            )
            match (variadic, single_block):
                case (False, False):
                    fields[name] = region_def()
                case (False, True):
                    fields[name] = region_def("single_block")
                case (True, False):
                    fields[name] = var_region_def()
                case (True, True):
                    fields[name] = var_region_def("single_block")
                case _:
                    pass  # Make pyright happy

        for [succ, name] in tblgen_op["successors"]["args"]:
            rec = self.js[succ["def"]]
            if "VariadicRegion" in rec["!superclasses"]:
                fields[name] = var_successor_def()
            else:
                fields[name] = successor_def()

        cls = type(
            tblgen_op["!name"],
            (IRDLOperation,),
            fields,
        )

        return irdl_op_definition(cls)


def tblgen_to_py(filename: str):
    js = json.load(open(filename))
    loader = TblgenLoader(js)
    dialects = js["!instanceof"]["Dialect"]
    for dialect in dialects:
        d = loader.load_dialect(dialect)
        print(f"Dialect {d.name}: attrs {list(d.attributes)}, ops {list(d.operations)}")


tblgen_to_py("test.json")
