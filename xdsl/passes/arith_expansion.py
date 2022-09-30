from __future__ import annotations
from xdsl.diagnostic import DiagnosticException
from xdsl.ir import MLContext
from xdsl.dialects.builtin import ModuleOp
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *

@dataclass(frozen=True)
class ExpandCeilDivUI(Strategy):
    # /// Expands CeilDivUIOp (n, m) into
    # ///  n == 0 ? 0 : ((n-1) / m) + 1
    # This rewrite is actually not 100% correct, as unsigned types are not supported currently
    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.CeilDivUI,
                        operands=[lhs, rhs],
                        results=[IResult(type)]):
                cst0 = new_cst_with_type(0, type)
                cmp = new_cmpi("eq", lhs, cst0)
                cst1 = new_cst_with_type(1, type)
                result = new_op(
                    arith.Select,
                    operands=[
                        cmp, cst0,
                        new_bin_op(
                            arith.Addi,
                            new_bin_op(arith.DivUI,
                                        new_bin_op(arith.Subi, lhs, cst1),
                                        rhs), cst1)
                    ],
                    result_types=[type])

                return success(cst0 + cmp + cst1 + result)
            case _:
                return failure(self)
        
@dataclass(frozen=True)
class ExpandCeilDivSI(Strategy):
    # /// Expands CeilDivSIOp (n, m) into
    # ///   1) x = (m > 0) ? -1 : 1
    # ///   2) (n*m>0) ? ((n+x) / m) + 1 : - (-n / m)

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.CeilDivSI,
                        operands=[lhs, rhs],
                        results=[IResult(type)]):
                cst_1 = new_cst_with_type(1, type)
                cst_0 = new_cst_with_type(0, type)
                cst_n1 = new_cst_with_type(-1, type)
                compare = new_cmpi("sgt", rhs, cst_0)
                # Compute x = (b>0) ? -1 : 1.
                x = new_op(arith.Select,
                            operands=[compare, cst_n1, cst_1],
                            result_types=[type])
                # Compute positive res: 1 + ((x+a)/b).
                xPlusLhs = new_bin_op(arith.Addi, x, lhs)
                xPlusLhsDivRhs = new_bin_op(arith.DivSI, xPlusLhs, rhs)
                posRes = new_bin_op(arith.Addi, cst_1, xPlusLhsDivRhs)
                # Compute negative res: - ((-a)/b).
                minusLhs = new_bin_op(arith.Subi, cst_0, lhs)
                minusLhsDivRhs = new_bin_op(arith.DivSI, minusLhs, rhs)
                negRes = new_bin_op(arith.Subi, cst_0, minusLhsDivRhs)
                # Do (a<0 && b<0) || (a>0 && b>0) instead of n*m to avoid overflow
                lhsNeg = new_cmpi("slt", lhs, cst_0)
                lhsPos = new_cmpi("sgt", lhs, cst_0)
                rhsNeg = new_cmpi("slt", rhs, cst_0)
                rhsPos = new_cmpi("sgt", rhs, cst_0)
                firstTerm = new_bin_op(arith.AndI, lhsNeg, rhsNeg)
                secondTerm = new_bin_op(arith.AndI, lhsPos, rhsPos)
                compareRes = new_bin_op(arith.OrI, firstTerm, secondTerm)
                result = new_op(arith.Select,
                                operands=[compareRes, posRes, negRes],
                                result_types=[type])
                # The order does not matter here but we choose the order that mlir-opt 
                # uses so we can compare the outputs better.
                return success(cst_1 + cst_0 + cst_n1 + compare + x + xPlusLhs + 
                               xPlusLhsDivRhs + posRes + minusLhs + minusLhsDivRhs + 
                               negRes + lhsNeg + lhsPos+ rhsNeg + rhsPos + firstTerm + 
                               secondTerm + compareRes + result)
            case _:
                return failure(self)

@dataclass(frozen=True)
class ExpandFloorDivSI(Strategy):
    # /// Expands FloorDivSIOp (n, m) into
    # ///   1)  x = (m<0) ? 1 : -1
    # ///   2)  return (n*m<0) ? - ((-n+x) / m) -1 : n / m

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.FloorDivSI,
                        operands=[lhs, rhs],
                        results=[IResult(type)]):
                cst_1 = new_cst_with_type(1, type)
                cst_0 = new_cst_with_type(0, type)
                cst_n1 = new_cst_with_type(-1, type)
                # Compute x = (b<0) ? 1 : -1.
                compare = new_cmpi("slt", rhs, cst_0)
                x = new_op(arith.Select,
                            operands=[compare, cst_1, cst_n1],
                            result_types=[type])
                # Compute negative res: -1 - ((x-a)/b).
                xMinusLhs = new_bin_op(arith.Subi, x, lhs)
                xMinusLhsDivRhs = new_bin_op(arith.DivSI, xMinusLhs, rhs)
                negRes = new_bin_op(arith.Subi, cst_n1, xMinusLhsDivRhs)
                # Compute positive res: a/b.
                posRes = new_bin_op(arith.DivSI, lhs, rhs)
                # Compute (a>0 && b<0) || (a>0 && b<0) instead of n*m<0
                lhsNeg = new_cmpi("slt", lhs, cst_0)
                lhsPos = new_cmpi("sgt", lhs, cst_0)
                rhsNeg = new_cmpi("slt", rhs, cst_0)
                rhsPos = new_cmpi("sgt", rhs, cst_0)
                firstTerm = new_bin_op(arith.AndI, lhsNeg, rhsPos)
                secondTerm = new_bin_op(arith.AndI, lhsPos, rhsNeg)
                compareRes = new_bin_op(arith.OrI, firstTerm, secondTerm)
                result = new_op(arith.Select,
                                operands=[compareRes, negRes, posRes],
                                result_types=[type])

                return success(cst_1 + cst_0 + cst_n1 + compare + x + xMinusLhs + 
                               xMinusLhsDivRhs + negRes + posRes + negRes + lhsNeg + 
                               lhsPos + rhsNeg + rhsPos + firstTerm + 
                               secondTerm + compareRes + result)
            case _:
                return failure(self)

class ExpandMaxMinI(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.MinUI | arith.MinSI | arith.MaxUI | arith.MaxSI,
                     operands=[lhs, rhs],
                     results=[IResult(type)]):
                match op.op_type:
                    case arith.MinUI:
                        pred = "ult"
                    case arith.MinSI:
                        pred = "slt"
                    case arith.MaxUI:
                        pred = "ugt"
                    case arith.MaxSI:
                        pred = "sgt"
                    case _:
                        return failure(self)
                cmp = new_cmpi(pred, lhs, rhs)
                result = new_op(arith.Select,
                                operands=[cmp, lhs, rhs],
                                result_types=[type])
                return success(cmp + result)
            case _:
                return failure(self)

class ExpandMaxMinF(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.Maxf | arith.Minf,
                     operands=[lhs, rhs],
                     results=[IResult(type)]):
                match op.op_type:
                    case arith.Maxf:
                        pred = "ugt"
                    case arith.Minf:
                        pred = "ult"
                    case _:
                        return failure(self)
                cmp = new_cmpf(pred, lhs, rhs)
                select = new_op(arith.Select,
                                operands=[cmp, lhs, rhs],
                                result_types=[type])
                # Handle the case where rhs is NaN: 'isNaN(rhs) ? rhs : select'.
                isNaN = new_cmpf("uno", rhs, rhs)
                result = new_op(arith.Select,
                                operands=[isNaN, rhs, select],
                                result_types=[type])
                return success(cmp + select + isNaN + result)
            case _:
                return failure(self)

def arith_expansion_pass(context: MLContext, module: ModuleOp) -> None:
    """Expand arithmetic operations."""

    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)

    imm_module: IOp = get_immutable_copy(module)

    arith_exand_strategy = everywhere(ExpandCeilDivUI()) ^ everywhere(ExpandCeilDivSI()) ^ \
        everywhere(ExpandFloorDivSI()) ^ everywhere(ExpandMaxMinI()) ^ everywhere(ExpandMaxMinF())
    rr = arith_exand_strategy.apply(imm_module)
    
    if not rr.isSuccess():
        # Should never happen as we don't use strategies that can fail.
        raise DiagnosticException("Strategy failed")

    if isinstance(new_module := rr.result_op.get_mutable_copy(), ModuleOp):
        module.regions = new_module.regions
    else:
        raise DiagnosticException("Unable to get mutable copy of new module")
