from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import SymbolTable


@dataclass(frozen=True)
class EqsatOptimizePDLInterp(ModulePass):
    name = "eqsat-optimize-pdl-interp"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        matcher = SymbolTable.lookup_symbol(op, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        mo = MatcherOptimizer(matcher)

        for gdo in matcher.walk():
            if isinstance(gdo, pdl_interp.GetDefiningOpOp):
                mo.optimize_gdo(gdo)


@dataclass
class MatcherOptimizer:
    matcher: pdl_interp.FuncOp

    finalize_blocks: set[Block] = field(default_factory=set[Block])

    def optimize_gdo(
        self,
        gdo: pdl_interp.GetDefiningOpOp,
    ) -> bool:
        assert (gdo_block := gdo.parent_block())
        if not (
            isinstance(gdo_block.last_op, pdl_interp.IsNotNullOp)
            and self.is_finalize_block(gdo_block.last_op.false_dest)
        ):
            return False

        potential_check_blocks = set[Block]()

        for use in gdo.input_op.uses:
            check_op = use.operation
            if isinstance(check_op, pdl_interp.SwitchOperationNameOp):
                finalize_dest = check_op.default_dest
            elif isinstance(check_op, pdl_interp.CheckOperationNameOp):
                finalize_dest = check_op.false_dest
            else:
                continue
            if not self.is_finalize_block(finalize_dest):
                continue
            assert (check_block := check_op.parent_block())
            potential_check_blocks.add(check_block)

        # Search for a path without branches that leads to one of the potential
        # blocks containing an operation name check. If there is such a path that
        # contains additional GetDefiningOpOps, we duplicate the check operation.
        additional_gdo_on_path = False
        block = gdo_block
        while (block := self.single_branch(block)) is not None:
            for op in block.ops:
                if isinstance(op, pdl_interp.GetDefiningOpOp):
                    additional_gdo_on_path = True
            if additional_gdo_on_path and block in potential_check_blocks:
                assert isinstance(
                    block.last_op,
                    pdl_interp.CheckOperationNameOp | pdl_interp.SwitchOperationNameOp,
                )
                self.duplicate_name_constraint(block.last_op, gdo_block)
                return True

        return False

    def is_finalize_block(self, block: Block) -> bool:
        if block in self.finalize_blocks:
            return True
        if len(block.ops) == 1 and isinstance(block.last_op, pdl_interp.FinalizeOp):
            self.finalize_blocks.add(block)
            return True
        return False

    def single_branch(self, block: Block) -> Block | None:
        """
        Check if the block has exactly one successor that is not a finalize block.
        If this is the case, we return the single successor. Otherwise, we return None.
        """
        assert block.last_op
        single_successor = None
        for succ in block.last_op.successors:
            if not self.is_finalize_block(succ):
                if single_successor is not None:
                    # At this point, we know that the block has more than one successor
                    return None
                single_successor = succ
        return single_successor

    def duplicate_name_constraint(
        self,
        check_op: pdl_interp.CheckOperationNameOp | pdl_interp.SwitchOperationNameOp,
        gdo_block: Block,
    ):
        # at this point, we also know that the false_dest of the terminator is a finalize block
        assert isinstance(terminator := gdo_block.last_op, pdl_interp.IsNotNullOp)

        continue_dest = terminator.true_dest
        if isinstance(check_op, pdl_interp.CheckOperationNameOp):
            new_check_op = pdl_interp.CheckOperationNameOp(
                check_op.operation_name,
                check_op.input_op,
                trueDest=continue_dest,
                falseDest=terminator.false_dest,
            )
        else:
            assert isinstance(check_op, pdl_interp.SwitchOperationNameOp)
            new_check_op = pdl_interp.SwitchOperationNameOp(
                check_op.case_values,
                check_op.input_op,
                default_dest=terminator.false_dest,
                cases=[continue_dest for _ in check_op.case_values],
            )
        new_block = Block((new_check_op,))
        self.matcher.body.insert_block_after(new_block, gdo_block)

        rewriter = Rewriter()
        rewriter.replace_op(
            terminator,
            pdl_interp.IsNotNullOp(
                terminator.value, trueDest=new_block, falseDest=terminator.false_dest
            ),
        )
