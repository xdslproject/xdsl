from __future__ import annotations
from re import L
from typing import Union
from enum import Enum

from idna import valid_contextj
from pyrsistent import optional

from xdsl.ir import *
from xdsl.util import new_op
from xdsl.irdl import *
from xdsl.parser import Parser
from xdsl.printer import Printer

@dataclass
class Affine:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(new_op("affine.for", 0, 0, 1))
        self.ctx.register_op(new_op("affine.load", 1, 3, 0))
        self.ctx.register_op(new_op("affine.store", 0, 4, 0))
        self.ctx.register_attr(AffineMap)

    def for_(self, lower_bound: int, upper_bound: int,
             block: Block) -> Operation:
        op = self.ctx.get_op("affine.for").create(
            [], [], regions=[Region.from_block_list([block])])
        return op

    def load(self, value: Union[Operation, SSAValue],
             i: Union[Operation, SSAValue], j: Union[Operation,
                                                     SSAValue]) -> Operation:
        return self.ctx.get_op("affine.load").create(
            [SSAValue.get(value),
             SSAValue.get(i),
             SSAValue.get(j)], [self.ctx.get_attr("f32")()], {})

    def store(self, value: Union[Operation, SSAValue],
              place: Union[Operation, SSAValue], i: Union[Operation, SSAValue],
              j: Union[Operation, SSAValue]) -> Operation:
        return self.ctx.get_op("affine.store").create([
            SSAValue.get(value),
            SSAValue.get(place),
            SSAValue.get(i),
            SSAValue.get(j)
        ], [], {})

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
        def get_affine_binary_expr(op: AffineOperation, lhs: AffineExpr, rhs: AffineExpr) -> AffineBinaryExpr:
            match op:
                case AffineOperation.neg:
                    return AffineBinaryExpr(AffineOperation.add, lhs, AffineBinaryExpr(AffineOperation.mul, AffineConstantExpr(-1), rhs))
                case _:
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
        
        def parse_affine_expr(llhs: optional[AffineExpr] = None, llhsOp: optional[AffineOperation] = None) -> AffineExpr:
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
        def parse_affine_high_prec_op_expr(llhs: optional[AffineExpr] = None, llhsOp: optional[AffineOperation] = None) -> AffineExpr:
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
    expr : list[AffineExpr]

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
        exprList = parser.parse_list(lambda : AffineExpr.parse(parser))
        parser.parse_char(")")

        return AffineMap(len(dimList), len(symbolList), exprList)
            
    def print(self, printer: Printer) -> None:
        dimString = ','.join(list(f'd{dimNum}' for dimNum in range(self.dimCount)))
        symbolString = ','.join(list(f's{symNum}' for symNum in range(self.symbolCount)))
        exprString = ','.join(list(f'{expr}' for expr in self.expr))
        printer.print_string(f'({dimString})[{symbolString}] -> ({exprString})')
        