from collections.abc import Collection
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
        mo.optimize()


@dataclass
class MatcherOptimizer:
    matcher: pdl_interp.FuncOp

    finalize_blocks: set[Block] = field(default_factory=set[Block])

    def optimize(self):
        for gdo in self.matcher.walk():
            if isinstance(gdo, pdl_interp.GetDefiningOpOp):
                self.optimize_name_constraints(gdo)

    def optimize_name_constraints(
        self,
        gdo: pdl_interp.GetDefiningOpOp,
    ) -> bool:
        assert (gdo_block := gdo.parent_block())
        if not (
            isinstance(gdo_block.last_op, pdl_interp.IsNotNullOp)
            and self.is_finalize_block(gdo_block.last_op.false_dest)
        ):
            return False

        check_blocks = set[Block]()
        names = set[builtin.StringAttr]()

        for use in gdo.input_op.uses:
            check_op = use.operation
            if isinstance(check_op, pdl_interp.SwitchOperationNameOp):
                finalizes = self.is_finalize_block(check_op.default_dest), f"{check_op}"
                names.update(check_op.case_values)
            elif isinstance(check_op, pdl_interp.CheckOperationNameOp):
                finalizes = self.is_finalize_block(check_op.false_dest)
                names.add(check_op.operation_name)
            else:
                continue
            if finalizes:
                # when this `check_op` is reached, the pattern will either fail to match or
                # the operation will be constrained to a name contained in `names`.
                assert (check_block := check_op.parent_block())
                check_blocks.add(check_block)

        worklist: list[Block] = list(gdo_block.last_op.successors)
        additional_gdo_on_path = False
        while worklist:
            block = worklist.pop()
            for op in block.ops:
                if isinstance(op, pdl_interp.GetDefiningOpOp):
                    additional_gdo_on_path = True
            if block in check_blocks:
                continue
            terminator = block.last_op
            assert terminator, "Expected each block to have a terminator"
            if isinstance(terminator, pdl_interp.RecordMatchOp):
                # There is a rewrite that is triggered without ever constraining the operation's name.
                # This means we cannot move a aggregate name check right after the GDO.
                return False
            worklist.extend(
                terminator.successors
            )  # This terminates since CFG is acyclic.

        # To prevent inserting too many new checks, we only insert if there is actual danger
        # of exponential blowup due to the presence of multiple GDOs on the path.
        if not additional_gdo_on_path:
            return False

        self.insert_name_constraint(
            names,
            gdo,
        )

        return True

    def is_finalize_block(self, block: Block) -> bool:
        if block in self.finalize_blocks:
            return True
        if len(block.ops) == 1 and isinstance(block.last_op, pdl_interp.FinalizeOp):
            self.finalize_blocks.add(block)
            return True
        return False

    def insert_name_constraint(
        self,
        valid_names: Collection[builtin.StringAttr],
        gdo: pdl_interp.GetDefiningOpOp,
    ):
        gdo_block = gdo.parent_block()
        # at this point, we also know that the false_dest of the terminator is a finalize block
        assert gdo_block
        assert isinstance(terminator := gdo_block.last_op, pdl_interp.IsNotNullOp)

        continue_dest = terminator.true_dest
        if len(valid_names) == 1:
            new_check_op = pdl_interp.CheckOperationNameOp(
                next(iter(valid_names)),
                gdo.input_op,
                trueDest=continue_dest,
                falseDest=terminator.false_dest,
            )
        else:
            new_check_op = pdl_interp.SwitchOperationNameOp(
                valid_names,
                gdo.input_op,
                default_dest=terminator.false_dest,
                cases=[continue_dest for _ in range(len(valid_names))],
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
