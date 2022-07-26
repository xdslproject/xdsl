from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.stencil as stencil
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
import difflib


def parse(program: str):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    stencil.Stencil(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()

    printer = Printer()
    printer.print_op(module)


def apply_strategy_and_compare(program: str, expected_program: str,
                               strategy: Strategy):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)
    stencil.Stencil(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()
    imm_module: IOp = get_immutable_copy(module)

    rr = strategy.apply(imm_module)
    assert rr.isSuccess()

    # for debugging
    printer = Printer()
    print(f'Result after applying "{strategy}":')
    printer.print_op(rr.result_op.get_mutable_copy())
    print()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(rr.result_op.get_mutable_copy())

    diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                    expected_program.splitlines(True))
    if file.getvalue().strip() != expected_program.strip():
        print("Did not get expected output! Diff:")
        print(''.join(diff))
        assert False


@dataclass(frozen=True)
class InlineProducer(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        # We match the consumer, rather than the producer.
        match op:
            case IOp(op_type=stencil.Apply,
                    operands=[ISSAValue(op=IOp(op_type=stencil.Apply) as producer_apply) as producer_result, *_] | [*_, ISSAValue(op=IOp(op_type=stencil.Apply) as producer_apply) as producer_result]):
                assert op.region is not None
                assert op.region.block is not None
                assert producer_apply.region is not None
                assert producer_apply.region.block is not None

                # Check that there are no stencil.store ops in the producer.
                for nested_op in op.region.ops:
                    if nested_op.op_type == stencil.Store and len(nested_op.operands) == 0:
                        return failure(self)
                        
                # Check that there is no stencil.dynAccess in the consumer accessing the producer 
                # (i.e. the blockArg associated with the producer)
                producer_idx = op.operands.index(producer_result)
                for nested_op in producer_apply.region.ops:
                    if nested_op.op_type == stencil.DynAccess and op.region.block.args[producer_idx] in nested_op.operands:
                        return failure(self)

                # Checking done, rewrite is possible
                consumer_lb = op.attributes["lb"]
                consumer_ub = op.attributes["ub"]

                producer_lb = producer_apply.attributes["lb"]
                producer_ub = producer_apply.attributes["ub"]

                # They just merge the args and canonicalize them away later. We will not do this!
                # i.e check before creation of the op, whether the args are actually used inside the region.
                env: dict[ISSAValue, ISSAValue] = {}

                # Create operands and blockArgs of the new ApplyOp:
                #   Merge operands of producer and consumer, no duplicate operands, and leave out the producer itself.
                #   Add a mapping of the old blockArgs to the new blockArgs.
                new_apply_operands: list[ISSAValue] = []
                new_apply_block_args: list[IBlockArg] = []
                for idx, operand in enumerate(producer_apply.operands + op.operands):
                    def add_mapping(index: int, block_arg: ISSAValue):
                        if index < len(producer_apply.operands):
                            env[producer_apply.region.block.args[index]] = block_arg
                        else: 
                            env[op.region.block.args[idx - len(producer_apply.operands)]] = block_arg

                    # Assuming no duplicated operands on the indicidual applys
                    # i.e. if there is a duplicate, the already added mapping is from the blockArg of the producer 
                    if operand in new_apply_operands:
                        block_arg = producer_apply.region.block.args[producer_apply.operands.index(operand)]
                        add_mapping(idx, env[block_arg])
                        continue
                    if operand in producer_apply.results:
                        # Will be inlined, not needed as operand
                        continue
                    new_apply_operands.append(operand)

                    # the block will set the block attribute here when it is created
                    new_apply_block_args.append(IBlockArg(typ=operand.typ, block=None, index=len(new_apply_block_args)))
                    add_mapping(idx, new_apply_block_args[-1])



                def get_ops_to_inline(old_ops: list[IOp], env: dict[ISSAValue, ISSAValue]) -> list[IOp]:
                    new_ops: list[IOp] = []

                    for old_op in old_ops:
                        if old_op.op_type == stencil.StoreResult:
                            # StoreResult denotes which value we have to use to replace the access op
                            assert access_op.result is not None
                            # In the inlined ops we move all references to the storeresult to its operand
                            env[old_op.result] = env[old_op.operands[0]]

                            # don't inline storeresult ops themselves
                            continue
                        elif old_op.op_type == stencil.Return:
                            # stencil.Return denotes which value we have to use to replace the access op
                            # The operand of the returnop has already been visited and inlined. We look it up in env
                            env[access_op.result] = env[old_op.operands[0]]

                            # don't inline the return op of the producer
                            continue
                        elif old_op.op_type == scf.If:
                            # rebuild the scf.If op
                            then_region_ops: list[IOp] = get_ops_to_inline(old_op.regions[0].ops, env=env)
                            else_region_ops: list[IOp] = get_ops_to_inline(old_op.regions[1].ops, env=env)
                            updated_scf_result_type = old_op.result_types[0].elem if isinstance(old_op.result_types[0], stencil.ResultType) else old_op.result_types[0]
                            new_ops.extend(from_op(old_op, regions=[IRegion([IBlock([], then_region_ops)]), IRegion([IBlock([], else_region_ops)])], env=env, result_types=[updated_scf_result_type]))
                            continue
                        # Ordinary inlining:
                        # Shift the producer op by the offset of the access op, i.e just adding the offsets
                        if "offset" in old_op.attributes.keys():
                            new_attributes = old_op.attributes.copy()

                            accessop_offset: list[int] = [int_attr.value.data for int_attr in access_op.attributes["offset"].data]
                            old_op_offset: list[int] = [int_attr.value.data for int_attr in new_attributes["offset"].data]

                            new_attributes["offset"] = ArrayAttr.from_list([IntegerAttr.from_int_and_width(accessop_offset[idx] + old_op_offset[idx], 64) for idx in range(3)])
                            new_ops.extend(from_op(old_op, env=env, attributes=new_attributes))
                        else:
                            new_ops.extend(from_op(old_op, env=env))                    
                    return new_ops

                # Create the region for the new ApplyOp
                new_apply_region_ops: list[IOp] = []
                for consumer_op in op.region.ops:
                    if (access_op := consumer_op).op_type == stencil.Access:
                        if op.operands[(accessed_block_arg_index := op.region.block.args.index(access_op.operands[0]))] in producer_apply.results:
                            # found an access op where I can do inlining!

                            # TODO: look in inlining cache for when the producer to be inlined has already been inlined
                            # We can do the inlining cache just with the env. So check here whether the access thingy is already in the env.

                            new_apply_region_ops.extend(get_ops_to_inline(producer_apply.region.ops, env=env))

                            # don't add the access op we do inlining for
                            continue

                    new_apply_region_ops.extend(from_op(consumer_op, env=env))

                # check whether all operands are actually used inside the region and remove duplicates


                new_apply = new_op(op_type=stencil.Apply, operands=new_apply_operands, 
                        result_types=op.result_types, attributes={"lb" : consumer_lb, "ub" : consumer_ub}, 
                        regions=[IRegion([IBlock(new_apply_block_args, new_apply_region_ops)])])

                return success(new_apply)
            case _:
                return failure(self)




def test_parse_inlining_simple():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !stencil.field<[70,70,70]>, %arg1 : !stencil.field<[70,70,70]>):
    %1 : !stencil.temp<[66,66,63]> = "stencil.load"(%arg0 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [66, 66, 63]]
    %2 : !stencil.temp<[64,64,60]>  = stencil.apply(%1 : !stencil.temp<[66,66,63]>) ["lb" = [1, 2, 3], "ub" = [65, 66, 63]] {
        ^bb0(%arg2: !stencil.temp<[66,66,63]>): 
        %3 : !f64 = stencil.access(%arg2: !stencil.temp<[66,66,63]>) ["offset" = [-1, 0, 0]]
        %4 : !f64 = stencil.access(%arg2: !stencil.temp<[66,66,63]>) ["offset" = [1, 0, 0]]
        %5 : !f64 = arith.addf(%3: !f64, %4: !f64)
        %6 : !stencil.result<!f64> = stencil.store_result(%5: !f64)
        stencil.return(%6: !stencil.result<!f64>)
    }
    %7 : !stencil.temp<[64,64,60]> = stencil.apply(%1 : !stencil.temp<[66,66,63]>, %2 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^bb0(%arg3: !stencil.temp<[66,66,63]>, %arg4: !stencil.temp<[64,64,60]>):
        %8 : !f64 = stencil.access(%arg3: !stencil.temp<[66,66,63]>) ["offset" = [0, 0, 0]]
        %9 : !f64 = stencil.access(%arg4: !stencil.temp<[64,64,60]>) ["offset" = [1, 2, 3]]
        %10 : !f64 = arith.addf(%8: !f64, %9: !f64)
        %11 : !stencil.result<!f64> = stencil.store_result(%10: !f64)
        stencil.return(%11: !stencil.result<!f64>)
    }
    stencil.store(%7 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70,70,70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]> = stencil.load(%0 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [66 : !i64, 66 : !i64, 63 : !i64]]
  %3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%2 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>):
    %5 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 0 : !i64, 0 : !i64]]
    %6 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [0 : !i64, 2 : !i64, 3 : !i64]]
    %7 : !f64 = stencil.access(%4 : !stencil.temp<[66 : !i64, 66 : !i64, 63 : !i64]>) ["offset" = [2 : !i64, 2 : !i64, 3 : !i64]]
    %8 : !f64 = arith.addf(%6 : !f64, %7 : !f64)
    %9 : !f64 = arith.addf(%5 : !f64, %8 : !f64)
    %10 : !stencil.result<!f64> = stencil.store_result(%9 : !f64)
    stencil.return(%10 : !stencil.result<!f64>)
  }
  stencil.store(%3 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    #  %0 = "stencil.cast"(%arg0) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %2 = "stencil.load"(%0) {lb = [0, 0, 0], ub = [66, 66, 63]} : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x66x63xf64>
    #     %3 = "stencil.apply"(%2) ( {
    #     ^bb0(%arg2: !stencil.temp<66x66x63xf64>):  // no predecessors

    #       %5 = "stencil.access"(%arg2) {offset = [-1, 0, 0]} : (!stencil.temp<66x66x63xf64>) -> f64
    #       %6 = "stencil.access"(%arg2) {offset = [1, 0, 0]} : (!stencil.temp<66x66x63xf64>) -> f64
    #       %7 = "std.addf"(%5, %6) : (f64, f64) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #     }) {lb = [1, 2, 3], ub = [65, 66, 63]} : (!stencil.temp<66x66x63xf64>) -> !stencil.temp<64x64x60xf64>
    #     %4 = "stencil.apply"(%2, %3) ( {
    #     ^bb0(%arg2: !stencil.temp<66x66x63xf64>, %arg3: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %5 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<66x66x63xf64>) -> f64
    #       %6 = "stencil.access"(%arg3) {offset = [1, 2, 3]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %7 = "std.addf"(%5, %6) : (f64, f64) -> f64
    #       %8 = "stencil.store_result"(%7) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%8) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<66x66x63xf64>, !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%4, %1) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   })

    # Source after:
    # //   func @simple(%arg0: !stencil.field<?x?x?xf64>, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg0([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %2 = stencil.load %0([0, 0, 0] : [66, 66, 63]) : (!stencil.field<70x70x70xf64>) -> !stencil.temp<66x66x63xf64>
    # //     %3 = stencil.apply (%arg2 = %2 : !stencil.temp<66x66x63xf64>) -> !stencil.temp<64x64x60xf64> {
    # //       %4 = stencil.access %arg2 [0, 0, 0] : (!stencil.temp<66x66x63xf64>) -> f64
    # //       %5 = stencil.access %arg2 [0, 2, 3] : (!stencil.temp<66x66x63xf64>) -> f64
    # //       %6 = stencil.access %arg2 [2, 2, 3] : (!stencil.temp<66x66x63xf64>) -> f64
    # //       %7 = addf %5, %6 : f64
    # //       %8 = addf %4, %7 : f64
    # //       %9 = stencil.store_result %8 : (f64) -> !stencil.result<f64>
    # //       stencil.return %9 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %3 to %1([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(before, after, seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))


def test_parse_inlining_simple_index():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !f64, %arg1 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[64,64,60]>  = stencil.apply(%arg0 : !f64) ["lb" = [1, 2, 3], "ub" = [65, 66, 63]] {
        ^0(%arg2: !f64): 
        %2 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [2, -1, 1]]
        %3 : !index = arith.constant() ["value" = 20 : !index]
        %4 : !f64 = arith.constant() ["value" = 0 : !i32]
        %5 : !i1 = arith.cmpi(%2 : !index, %3 : !index) ["predicate" = 2 : !i64]
        %6 : !f64 = arith.select(%5 : !i1, %arg2 : !f64, %4: !f64)
        %7 : !stencil.result<!f64> = stencil.store_result(%6 : !f64)
        stencil.return(%7 : !stencil.result<!f64>)
    }
    %8 : !stencil.temp<[64,64,60]> = stencil.apply(%arg0 : !f64, %1 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg3: !f64, %arg4: !stencil.temp<[64,64,60]>):
        %9 : !i64 = stencil.access(%arg4: !stencil.temp<[64,64,60]>) ["offset" = [1, 2, 3]]
        %10 : !f64 = arith.addf(%9: !i64, %arg3: !f64)
        %11 : !stencil.result<!f64> = stencil.store_result(%10: !f64)
        stencil.return(%11 : !stencil.result<!f64>)
    }
    stencil.store(%8: !stencil.temp<[64, 64, 60]>, %arg1: !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !f64, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%0 : !f64) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%3 : !f64):
    %4 : !index = stencil.index() ["dim" = 2 : !i64, "offset" = [3 : !i64, 1 : !i64, 4 : !i64]]
    %5 : !index = arith.constant() ["value" = 20 : !index]
    %6 : !f64 = arith.constant() ["value" = 0 : !i32]
    %7 : !i1 = arith.cmpi(%4 : !index, %5 : !index) ["predicate" = 2 : !i64]
    %8 : !f64 = arith.select(%7 : !i1, %3 : !f64, %6 : !f64)
    %9 : !f64 = arith.addf(%8 : !f64, %3 : !f64)
    %10 : !stencil.result<!f64> = stencil.store_result(%9 : !f64)
    stencil.return(%10 : !stencil.result<!f64>)
  }
  stencil.store(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    # "func"() ( {
    #   ^bb0(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.apply"(%arg0) ( {
    #     ^bb0(%arg2: f64):  // no predecessors
    #       %3 = "stencil.index"() {dim = 2 : i64, offset = [2, -1, 1]} : () -> index
    #       %c20 = "std.constant"() {value = 20 : index} : () -> index
    #       %cst = "std.constant"() {value = 0.000000e+00 : f64} : () -> f64
    #       %4 = "std.cmpi"(%3, %c20) {predicate = 2 : i64} : (index, index) -> i1
    #       %5 = "std.select"(%4, %arg2, %cst) : (i1, f64, f64) -> f64
    #       %6 = "stencil.store_result"(%5) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%6) : (!stencil.result<f64>) -> ()
    #     }) {lb = [1, 2, 3], ub = [65, 66, 63]} : (f64) -> !stencil.temp<64x64x60xf64>
    #     %2 = "stencil.apply"(%arg0, %1) ( {
    #     ^bb0(%arg2: f64, %arg3: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %3 = "stencil.access"(%arg3) {offset = [1, 2, 3]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %4 = "std.addf"(%3, %arg2) : (f64, f64) -> f64
    #       %5 = "stencil.store_result"(%4) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%5) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (f64, !stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%2, %0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "simple_index", type = (f64, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    

    # Source after:
    # // module  {
    # //   func @simple_index(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<64x64x60xf64> {
    # //       %c20 = constant 20 : index
    # //       %cst = constant 0.000000e+00 : f64
    # //       %2 = stencil.index 2 [3, 1, 4] : index
    # //       %3 = cmpi slt, %2, %c20 : index
    # //       %4 = select %3, %arg2, %cst : f64
    # //       %5 = addf %4, %arg2 : f64
    # //       %6 = stencil.store_result %5 : (f64) -> !stencil.result<f64>
    # //       stencil.return %6 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %1 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(before, after, seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))



def test_parse_inlining_simple_ifelse():
    before = \
"""
  func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
  ^0(%arg0 : !f64, %arg1 : !stencil.field<[70, 70, 70]>):
    %1 : !stencil.temp<[64,64,60]> = stencil.apply(%arg0 : !f64) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^0(%arg2: !f64): 
        %true : !i1 = arith.constant() ["value" = 1 : !i1]
        %2 : !stencil.result<!f64> = scf.if(%true : !i1) {
            %3 : !stencil.result<!f64> = stencil.store_result(%arg2 : !f64)
            scf.yield(%3 : !stencil.result<!f64>)
        } {
            %4 : !stencil.result<!f64> = stencil.store_result(%arg2 : !f64)
            scf.yield(%4 : !stencil.result<!f64>)
        }
        stencil.return(%2 : !stencil.result<!f64>)
    }
    %5 : !stencil.temp<[64,64,60]> = stencil.apply(%1 : !stencil.temp<[64,64,60]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]] {
        ^1(%arg3: !stencil.temp<[64,64,60]>):
        %6 : !f64 = stencil.access(%arg3: !stencil.temp<[64,64,60]>) ["offset" = [0, 0, 0]]
        %7 : !stencil.result<!f64> = stencil.store_result(%6 : !f64)
        stencil.return(%7 : !stencil.result<!f64>)
    }
    stencil.store(%5 : !stencil.temp<[64,64,60]>, %arg1 : !stencil.field<[70, 70, 70]>) ["lb" = [0, 0, 0], "ub" = [64, 64, 60]]
    func.return()
  }
"""

    after  = \
"""
func.func() ["sym_name" = "test", "type" = !fun<[], []>] {
^0(%0 : !f64, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>):
  %2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]> = stencil.apply(%0 : !f64) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]] {
  ^1(%3 : !f64):
    %4 : !i1 = arith.constant() ["value" = 1 : !i1]
    %5 : !f64 = scf.if(%4 : !i1) {
      scf.yield(%3 : !f64)
    } {
      scf.yield(%3 : !f64)
    }
    %6 : !stencil.result<!f64> = stencil.store_result(%5 : !f64)
    stencil.return(%6 : !stencil.result<!f64>)
  }
  stencil.store(%2 : !stencil.temp<[64 : !i64, 64 : !i64, 60 : !i64]>, %1 : !stencil.field<[70 : !i64, 70 : !i64, 70 : !i64]>) ["lb" = [0 : !i64, 0 : !i64, 0 : !i64], "ub" = [64 : !i64, 64 : !i64, 60 : !i64]]
  func.return()
}
"""
    # Source before:
    #   "func"() ( {
    #   ^bb0(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>):  // no predecessors
    #     %0 = "stencil.cast"(%arg1) {lb = [-3, -3, -3], ub = [67, 67, 67]} : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    #     %1 = "stencil.apply"(%arg0) ( {
    #     ^bb0(%arg2: f64):  // no predecessors
    #       %true = "std.constant"() {value = true} : () -> i1
    #       %3 = "scf.if"(%true) ( {
    #         %4 = "stencil.store_result"(%arg2) : (f64) -> !stencil.result<f64>
    #         "scf.yield"(%4) : (!stencil.result<f64>) -> ()
    #       },  {
    #         %4 = "stencil.store_result"(%arg2) : (f64) -> !stencil.result<f64>
    #         "scf.yield"(%4) : (!stencil.result<f64>) -> ()
    #       }) : (i1) -> !stencil.result<f64>
    #       "stencil.return"(%3) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (f64) -> !stencil.temp<64x64x60xf64>
    #     %2 = "stencil.apply"(%1) ( {
    #     ^bb0(%arg2: !stencil.temp<64x64x60xf64>):  // no predecessors
    #       %3 = "stencil.access"(%arg2) {offset = [0, 0, 0]} : (!stencil.temp<64x64x60xf64>) -> f64
    #       %4 = "stencil.store_result"(%3) : (f64) -> !stencil.result<f64>
    #       "stencil.return"(%4) : (!stencil.result<f64>) -> ()
    #     }) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>) -> !stencil.temp<64x64x60xf64>
    #     "stencil.store"(%2, %0) {lb = [0, 0, 0], ub = [64, 64, 60]} : (!stencil.temp<64x64x60xf64>, !stencil.field<70x70x70xf64>) -> ()
    #     "std.return"() : () -> ()
    #   }) {stencil.program, sym_name = "simple_ifelse", type = (f64, !stencil.field<?x?x?xf64>) -> ()} : () -> ()
    
    

    # Source after:
    # // module  {
    # //   func @simple_ifelse(%arg0: f64, %arg1: !stencil.field<?x?x?xf64>) attributes {stencil.program} {
    # //     %0 = stencil.cast %arg1([-3, -3, -3] : [67, 67, 67]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x70xf64>
    # //     %1 = stencil.apply (%arg2 = %arg0 : f64) -> !stencil.temp<64x64x60xf64> {
    # //       %true = constant true
    # //       %2 = scf.if %true -> (f64) {
    # //         scf.yield %arg2 : f64
    # //       } else {
    # //         scf.yield %arg2 : f64
    # //       }
    # //       %3 = stencil.store_result %2 : (f64) -> !stencil.result<f64>
    # //       stencil.return %3 : !stencil.result<f64>
    # //     } to ([0, 0, 0] : [64, 64, 60])
    # //     stencil.store %1 to %0([0, 0, 0] : [64, 64, 60]) : !stencil.temp<64x64x60xf64> to !stencil.field<70x70x70xf64>
    # //     return
    # //   }
    # // }

    apply_strategy_and_compare(before, after, seq(topToBottom(InlineProducer()), topToBottom(GarbageCollect())))




if __name__ == "__main__":
    test_parse_inlining_simple()
    test_parse_inlining_simple_index()
    test_parse_inlining_simple_ifelse()