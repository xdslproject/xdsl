from dataclasses import dataclass
from typing import List, Set

from xdsl.dialects import scf, symref
from xdsl.frontend.codegen.exception import prettify
from xdsl.ir import Operation, SSAValue, Region
from xdsl.rewriter import Rewriter


@dataclass
class PromoteException(Exception):
    """
    Exception type if something goes terribly wrong when running `desymref` pass.
    """
    op: Operation
    msg: str

    def __str__(self) -> str:
        return f"Unable to promote '{prettify(self.op)}': {self.msg}."


@dataclass
class ScfIfPromoter:
    """Class responsible for promoting scf.if in desymrification pass."""
    rewriter: Rewriter
    if_op: scf.If

    def check_promotable(self):
        num_true_blocks = len(self.if_op.true_region.blocks)
        num_false_blocks = len(self.if_op.false_region.blocks)
        if num_true_blocks > 1 or num_false_blocks > 1:
            raise PromoteException(self.if_op, f"found {num_true_blocks} true and {num_false_blocks} blocks, but expected 0 or 1 only")

    def promote_fetch_ops(self, symbol: str) -> None | symref.Fetch:
        true_block = self.if_op.true_region.blocks[0]
        false_block = self.if_op.false_region.blocks[0]

        # Find fetches of this symbol.
        fetch_ops: List[symref.Fetch] = []
        for op in true_block.ops:
            if isinstance(op, symref.Fetch):
                symbol_op = op.attributes["symbol"].data.data
                if symbol == symbol_op:
                    fetch_ops.append(op)
        for op in false_block.ops:
            if isinstance(op, symref.Fetch):
                symbol_op = op.attributes["symbol"].data.data
                if symbol == symbol_op:
                    fetch_ops.append(op)
        
        if len(fetch_ops) == 0:
            # This symbol was not fetched, we are done.
            return None
        else:
            # If this symbol is fetched, promote it outside and delete
            # fetches in both true and false blocks.
            if_parent_block = self.if_op.parent_block()
            if_idx = if_parent_block.get_operation_index(self.if_op)
            parent_fetch_op = fetch_ops[0].clone()
            if_parent_block.insert_op(parent_fetch_op, if_idx)
            for fetch_op in fetch_ops:
                self.rewriter.replace_op(fetch_op, [], [parent_fetch_op.results[0]])
            return parent_fetch_op

    def promote_update_ops(self, symbol: str, fetch_op: None | symref.Fetch):
        true_block = self.if_op.true_region.blocks[0]
        false_block = self.if_op.false_region.blocks[0]

        update_ty = None
        true_block_update_op = None
        false_block_update_op = None
        for op in true_block.ops:
            if isinstance(op, symref.Update):
                symbol_op = op.attributes["symbol"].data.data
                if symbol == symbol_op:
                    update_ty = op.operands[0].typ
                    true_block_update_op = op
        for op in false_block.ops:
            if isinstance(op, symref.Update):
                symbol_op = op.attributes["symbol"].data.data
                if symbol == symbol_op:
                    update_ty = op.operands[0].typ
                    false_block_update_op = op
        
        if true_block_update_op is None and false_block_update_op is None:
            return None

        # We have to add a fetch (if haven't done before)
        if fetch_op is None:
            if_parent_block = self.if_op.parent_block()
            if_idx = if_parent_block.get_operation_index(self.if_op)
            fetch_op = symref.Fetch.get(symbol, update_ty)
            if_parent_block.insert_op(fetch_op, if_idx)
        
        if true_block_update_op is not None:
            true_block_yeild = true_block_update_op.operands[0]
            self.rewriter.erase_op(true_block_update_op)
        else:
            true_block_yeild = fetch_op.results[0]

        if false_block_update_op is not None:
            false_block_yeild = false_block_update_op.operands[0]
            self.rewriter.erase_op(false_block_update_op)
        else:
            false_block_yeild = fetch_op.results[0]
        return true_block_yeild, false_block_yeild

    def promote(self, symbols: Set[str]):
        self.check_promotable()
        true_block = self.if_op.true_region.blocks[0]
        false_block = self.if_op.false_region.blocks[0]

        true_block_yield_ops = []
        false_block_yield_ops = []

        fetch_ops = []
        for symbol in symbols:
            fetch_op = self.promote_fetch_ops(symbol)
            fetch_ops.append(fetch_op)

        updated = []
        for i, symbol in enumerate(symbols): 
            res = self.promote_update_ops(symbol, fetch_ops[i])
            if res is not None:
                ty, fy = res
                updated.append(symbol)
                true_block_yield_ops.append(ty)
                false_block_yield_ops.append(fy)
        
        self.rewriter.replace_op(true_block.ops[-1], scf.Yield.get(*true_block_yield_ops))
        self.rewriter.replace_op(false_block.ops[-1], scf.Yield.get(*false_block_yield_ops))

        ret_types = list(map(lambda op: op.typ, true_block_yield_ops))

        new_true_region = Region()
        self.if_op.true_region.clone_into(new_true_region)
        new_false_region = Region()
        self.if_op.false_region.clone_into(new_false_region)

        if_parent_block = self.if_op.parent_block()
        if_idx = if_parent_block.get_operation_index(self.if_op)
        new_if_op = scf.If.get(self.if_op.cond, ret_types, new_true_region, new_false_region)
        if_parent_block.insert_op(new_if_op, if_idx)
        self.rewriter.erase_op(self.if_op)

        for i, symbol in enumerate(updated): 
            # make sure symbol is updated!
            pb = new_if_op.parent_block()
            if_idx = pb.get_operation_index(new_if_op)
            symbol_update_op = symref.Update.get(symbol, new_if_op.results[i])
            pb.insert_op(symbol_update_op, if_idx + 1)

        

        
