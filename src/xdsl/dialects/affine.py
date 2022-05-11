from __future__ import annotations
from typing import Union, Optional
from enum import Enum

from xdsl.ir import Operation, Region, SSAValue
from dataclasses import dataclass
from xdsl.dialects.builtin import IntegerAttr, IndexType
from xdsl.irdl import irdl_op_definition, irdl_attr_definition, AttributeDef, OperandDef, RegionDef, VarResultDef, VarOperandDef, AnyAttr, Data
from xdsl.parser import Parser
from xdsl.printer import Printer


@dataclass
class Affine:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(For)
        self.ctx.register_op(Yield)
        self.ctx.register_attr(AffineMap)


@irdl_op_definition
class For(Operation):
    name: str = "affine.for"
    arguments = VarOperandDef(AnyAttr())
    res = VarResultDef(AnyAttr())

    # TODO the bounds are in fact affine_maps
    # TODO support dynamic bounds as soon as maps are here
    lower_bound = AttributeDef(IntegerAttr)
    upper_bound = AttributeDef(IntegerAttr)
    step = AttributeDef(IntegerAttr)

    body = RegionDef()

    def verify_(self) -> None:
        if len(self.operands) != len(self.results):
            raise Exception("Expected the same amount of operands and results")

        operand_types = [SSAValue.get(op).typ for op in self.operands]
        if (operand_types != [res.typ for res in self.results]):
            raise Exception(
                "Expected all operands and result pairs to have matching types"
            )

        entry_block: Block = self.body.blocks[0]
        if ([IndexType()] + operand_types !=
            [arg.typ for arg in entry_block.args]):
            raise Exception(
                "Expected BlockArguments to have the same types as the operands"
            )

    @staticmethod
    def from_region(operands: List[Union[Operation, SSAValue]],
                    lower_bound: Union[int, IntegerAttr],
                    upper_bound: Union[int, IntegerAttr],
                    region: Region,
                    step: Union[int, IntegerAttr] = 1) -> For:
        result_types = [SSAValue.get(op).typ for op in operands]
        return For.build(operands=[[operand for operand in operands]],
                         result_types=[result_types],
                         attributes={
                             "lower_bound": lower_bound,
                             "upper_bound": upper_bound,
                             "step": step
                         },
                         regions=[region])

    @staticmethod
    def from_callable(operands: List[Union[Operation, SSAValue]],
                      lower_bound: Union[int, IntegerAttr],
                      upper_bound: Union[int, IntegerAttr],
                      body: Callable[[BlockArgument, ...], List[Operation]],
                      step: Union[int, IntegerAttr] = 1) -> For:
        arg_types = [IndexType()] + [SSAValue.get(op).typ for op in operands]
        return For.from_region(
            operands, lower_bound, upper_bound,
            Region.from_block_list([Block.from_callable(arg_types, body)]),
            step)


@irdl_op_definition
class Yield(Operation):
    name: str = "affine.yield"
    arguments = VarOperandDef(AnyAttr())

    @staticmethod
    def get(*operands: Union[Operation, SSAValue]) -> Yield:
        return Yield.create(operands=[operand for operand in operands])


class AffineOperation(Enum):
    add = '+'
    mul = '*'
    ceildiv = '/'
    floordiv = '//'
    mod = '%'
    neg = '-'


@dataclass
class AffineExpr:

    def __add__(self, other: int):
        return self + AffineConstantExpr(other)

    def __add__(self, otherExpr: AffineExpr):
        return AffineBinaryExpr(AffineOperation.add, self, otherExpr)

    def __sub__(self, other: int):
        return self - AffineConstantExpr(other)

    def __sub__(self, other: AffineExpr):
        return AffineBinaryExpr(AffineOperation.add, self, (other * -1))

    def __mul__(self, other: int):
        return self * AffineConstantExpr(other)

    def __mul__(self, other: AffineExpr):
        return AffineBinaryExpr(AffineOperation.mul, self, other)

    # used for ceildiv
    def __truediv__(self, other: int):
        return self / AffineConstantExpr(int)

    def __truediv__(self, other: AffineExpr):
        return AffineBinaryExpr(AffineOperation.ceildiv, self, other)

    def __floordiv__(self, other: int):
        return self // AffineConstantExpr(int)

    def __floordiv__(self, other: AffineExpr):
        return AffineBinaryExpr(AffineOperation.floordiv, self, other)

    def __mod__(self, other: int):
        return self % AffineConstantExpr(int)

    def __mod__(self, other: AffineExpr):
        return AffineBinaryExpr(AffineOperation.mod, self, other)

    def __neg__(self):
        return self * -1

    def parse(parser: Parser) -> AffineExpr:

        def get_affine_binary_expr(op: AffineOperation, lhs: AffineExpr,
                                   rhs: AffineExpr) -> AffineBinaryExpr:
            if op == AffineOperation.neg:
                return AffineBinaryExpr(
                    AffineOperation.add, lhs,
                    AffineBinaryExpr(AffineOperation.mul,
                                     AffineConstantExpr(-1), rhs))
            else:
                return AffineBinaryExpr(op, lhs, rhs)

        def parse_optional_affine_low_prec_op() -> AffineOperation:
            if parser.parse_optional_char('+'):
                return AffineOperation.add
            if parser.parse_optional_char('-'):
                return AffineOperation.neg

        def parse_optional_affine_high_prec_op() -> AffineOperation:
            if parser.parse_optional_char('*'):
                return AffineOperation.mul
            if parser.parse_optional_char('-'):
                return AffineOperation.neg
            if parser.parse_optional_char('/'):
                if parser.parse_optional_char('/'):
                    return AffineOperation.floordiv
                return AffineOperation.ceildiv
            if parser.parse_optional_char('%'):
                return AffineOperation.mod

        def parse_affine_operand() -> AffineExpr:
            if parser.parse_optional_char('d'):
                index = parser.parse_int_literal()
                return AffineDimExpr(index)
            if parser.parse_optional_char('s'):
                index = parser.parse_int_literal()
                return AffineSymbolExpr(index)
            constant: int = parser.parse_optional_int_literal()
            if constant:
                return AffineConstantExpr(constant)
            if parser.parse_optional_char('('):
                if parser.peek_char(')'):
                    raise Exception("no expression inside parentheses")
                expr = parse_affine_expr()
                parser.parse_char(')')
                return expr
            if parser.parse_optional_char('-'):
                expr = parse_affine_operand()
                return expr * -1

        def parse_affine_expr(
                llhs: Optional[AffineExpr] = None,
                llhsOp: Optional[AffineOperation] = None) -> AffineExpr:
            lhs = parse_affine_operand()
            if not lhs:
                return
            lOp = parse_optional_affine_low_prec_op()
            if lOp:
                if llhs:
                    expr = get_affine_binary_expr(llhsOp, llhs, lhs)
                    return parse_affine_expr(expr, lOp)
                # no llhs, get rhs and form the expression
                return parse_affine_expr(lhs, lOp)
            hOp = parse_optional_affine_high_prec_op()
            if hOp:
                # first evaluate higher order precedence expr:
                highExpr = parse_affine_high_prec_op_expr(lhs, hOp)
                if not highExpr:
                    return
                if llhs:
                    expr = get_affine_binary_expr(llhsOp, llhs, highExpr)
                else:
                    expr = highExpr
                nextOp = parse_optional_affine_low_prec_op()
                if nextOp:
                    return parse_affine_expr(expr, nextOp)
                return expr

            if llhs:
                return get_affine_binary_expr(llhsOp, llhs, lhs)
            return lhs

        # essentially the same structure as parsing parse_affine_expr but for higher precedence ops
        # this is mirrored to the approach in MLIR. TODO look whether the two functions could be combined
        def parse_affine_high_prec_op_expr(
                llhs: Optional[AffineExpr] = None,
                llhsOp: Optional[AffineOperation] = None) -> AffineExpr:
            lhs = parse_affine_operand()
            if not lhs:
                return
            op = parse_optional_affine_high_prec_op()
            if op:
                if llhs:
                    expr = get_affine_binary_expr(llhsOp, llhs, lhs)
                    return parse_affine_high_prec_op_expr(expr, op)
                return parse_affine_high_prec_op_expr(lhs, op)
            # no op, this is the last operand in this expression
            if llhs:
                return get_affine_binary_expr(llhsOp, llhs, lhs)
            return lhs

        if parser.peek_char(')'):
            return
        return parse_affine_expr()


@dataclass
class AffineDimExpr(AffineExpr):
    index: int

    def __str__(self) -> str:
        return 'd' + str(self.index)


@dataclass
class AffineSymbolExpr(AffineExpr):
    index: int

    def __str__(self) -> str:
        return 's' + str(self.index)


@dataclass
class AffineConstantExpr(AffineExpr):
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class AffineBinaryExpr(AffineExpr):
    operation: AffineOperation
    lhs: AffineExpr
    rhs: AffineExpr

    def __str__(self) -> str:
        return "(" + str(self.lhs) + self.operation.value + str(self.rhs) + ")"


@irdl_attr_definition
class AffineMap(Data):
    name = "affine_map"
    dimCount: int
    symbolCount: int
    expr: list[AffineExpr]

    @staticmethod
    def parse(parser: Parser) -> AffineMap:

        def parse_optional_affineDim() -> str:
            if parser.parse_optional_char("d"):
                return "d" + parser.parse_alpha_num()

        def parse_optional_affineSymbol() -> str:
            if parser.parse_optional_char("s"):
                return "s" + parser.parse_alpha_num()

        parser.parse_char("(")
        dimList = parser.parse_list(parse_optional_affineDim)
        parser.parse_char(")")

        parser.parse_char("[")
        symbolList = parser.parse_list(parse_optional_affineSymbol)
        parser.parse_char("]")

        parser.parse_char("-")
        parser.parse_char(">")

        parser.parse_char("(")
        exprList = parser.parse_list(lambda: AffineExpr.parse(parser))
        parser.parse_char(")")

        return AffineMap(len(dimList), len(symbolList), exprList)

    def print(self, printer: Printer) -> None:
        dimString = ','.join(
            list(f'd{dimNum}' for dimNum in range(self.dimCount)))
        symbolString = ','.join(
            list(f's{symNum}' for symNum in range(self.symbolCount)))
        exprString = ','.join(list(f'{expr}' for expr in self.expr))
        printer.print_string(
            f'({dimString})[{symbolString}] -> ({exprString})')
