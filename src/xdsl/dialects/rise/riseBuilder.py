from __future__ import annotations
from xdsl.dialects.arith import Arith
from xdsl.printer import Printer
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.memref import *
from xdsl.dialects.builtin import IntegerType, Float32Type, IntegerAttr, FlatSymbolRefAttr
from xdsl.dialects.rise.rise import *
from xdsl.dialects.std import Return as stdReturn
from xdsl.parser import Parser


@dataclass
class RiseBuilder:
    ctx: MLContext
    # insertion_point: Tuple[Block, int] = (None, 0)
    current_block: Block = None

    def _attach(self, op):
        if self.current_block is None:
            return
        self.current_block.insert_op(op, len(self.current_block.ops))

    def getSSAValue(self, opList) -> SSAValue:
        # last op in the list is the one we want. usually the apply
        if isinstance(opList, List):
            opList = opList[-1]
        return SSAValue.get(opList)

    def nat(self, value: int) -> NatAttr:
        return NatAttr.from_int(value)

    def array(self, size: Union[int, NatAttr], elemT: DataType):
        if isinstance(size, int):
            size = NatAttr.from_int(size)
        return ArrayType.from_length_and_elemT(size, elemT)

    def scalar(self, wrapped: Attribute):
        return ScalarType.from_wrapped_type(wrapped)

    def tuple(self, left: DataType, right: DataType):
        return TupleType.from_types(left, right)

    def fun(self, left: Union[RiseType, DataType], right: Union[RiseType,
                                                                DataType]):
        return FunType.from_types(left, right)

    def inOp(self, value: Union[Operation, SSAValue],
             type: DataType) -> Operation:
        op = In.create([self.getSSAValue(value)], [type], {"type": type})
        self._attach(op)
        return op

    def out(self, input: Union[Operation, SSAValue],
            output: Union[Operation, SSAValue]) -> Operation:
        op = Out.create([self.getSSAValue(input), self.getSSAValue(output)])
        self._attach(op)
        return op

    def apply(self, fun: Union[Operation, SSAValue],
              *args: Union[Operation, SSAValue]) -> Operation:
        op = Apply.create(
            [self.getSSAValue(fun), *[self.getSSAValue(arg) for arg in args]],
            [self.getSSAValue(fun).typ.get_output_recursive()])
        self._attach(op)
        return op

    def zip(self, left: Union[Operation, SSAValue],
            right: Union[Operation, SSAValue]) -> Operation:
        left = self.getSSAValue(left)
        right = self.getSSAValue(right)

        assert (isinstance(left.typ, ArrayType)
                & isinstance(right.typ, ArrayType))

        assert (left.typ.size == right.typ.size)
        n = left.typ.size
        s = left.typ.elemType
        t = right.typ.elemType

        zip = Zip.create([],
                         result_types=[
                             self.fun(
                                 self.array(n, s),
                                 self.fun(self.array(n, t),
                                          self.array(n, self.tuple(s, t))))
                         ],
                         attributes={
                             "n": n,
                             "s": s,
                             "t": t
                         })
        self._attach(zip)
        apply = self.apply(zip, left, right)
        return [zip, apply]

    def tupleOp(self, left: Union[Operation, SSAValue],
                right: Union[Operation, SSAValue]) -> list(Operation):
        left = self.getSSAValue(left)
        right = self.getSSAValue(right)
        assert (isinstance(left.typ, ArrayType)
                & isinstance(right.typ, ArrayType))
        s = left.typ.elemType
        t = right.typ.elemType

        tuple = Tuple.create(
            [],
            result_types=[self.fun(s, self.fun(t, self.tuple(s, t)))],
            attributes={
                "s": s,
                "t": t
            })
        self._attach(tuple)
        apply = self.apply(tuple, left, right)
        return [tuple, apply]

    def fst(self, value: Union[Operation, SSAValue]) -> list(Operation):
        assert (isinstance(value.typ, TupleType))
        value = self.getSSAValue(value)
        s = value.typ.left
        t = value.typ.right
        fst = Fst.create([],
                         result_types=[self.fun(self.tuple(s, t), s)],
                         attributes={
                             "s": s,
                             "t": t
                         })
        self._attach(fst)
        apply = self.apply(fst, value)
        return [fst, apply]

    def snd(self, value: Union[Operation, SSAValue]) -> list(Operation):
        assert (isinstance(value.typ, TupleType))
        value = self.getSSAValue(value)
        s = value.typ.left
        t = value.typ.right
        snd = Snd.create([],
                         result_types=[self.fun(self.tuple(s, t), t)],
                         attributes={
                             "s": s,
                             "t": t
                         })
        self._attach(snd)
        apply = self.apply(snd, value)
        return [snd, apply]

    def map(self, _lambda: Union[Operation, SSAValue],
            array: Union[Operation, SSAValue]) -> list(Operation):
        _lambda = self.getSSAValue(_lambda)
        array = self.getSSAValue(array)
        assert (isinstance(array.typ, ArrayType))
        n = array.typ.size
        s = array.typ.elemType
        lambdaOutputType = _lambda.typ.get_output_recursive()
        assert (isinstance(lambdaOutputType, ArrayType))
        t = lambdaOutputType.elemType
        map = Map.create([],
                         result_types=[
                             self.fun(
                                 self.fun(s, t),
                                 self.fun(self.array(n, s), self.array(n, t)))
                         ],
                         attributes={
                             "n": n,
                             "s": s,
                             "t": t
                         })
        self._attach(map)
        apply = self.apply(map, _lambda, array)
        return [_lambda, map, apply]

    def reduce(
        self, init: Union[Operation, SSAValue],
        array: Union[Operation, SSAValue], lambda_arg_types: List[Attribute],
        body: Callable[[BlockArgument, ...],
                       List[Operation]]) -> list(Operation):
        init = self.getSSAValue(init)
        array = self.getSSAValue(array)
        assert (isinstance(array.typ, ArrayType))
        n = array.typ.size
        s = array.typ.elemType
        _lambda = self._lambda(lambda_arg_types, body)
        t = self.getSSAValue(_lambda).typ.get_output_recursive()
        reduce = Reduce.create([],
                               result_types=[
                                   self.fun(
                                       self.fun(s, self.fun(t, t)),
                                       self.fun(t,
                                                self.fun(self.array(n, s), t)))
                               ],
                               attributes={
                                   "n": n,
                                   "s": s,
                                   "t": t
                               })
        self._attach(reduce)
        apply = self.apply(reduce, _lambda, init, array)
        return [_lambda, reduce, apply]

    def _lambda(
            self, lambda_arg_types: List[Attribute],
            body: Callable[[BlockArgument, ...],
                           List[Operation]]) -> Operation:
        saveCurrentBlock = self.current_block
        block = Block.from_arg_types(lambda_arg_types)
        self.current_block = block
        body(*block.args)
        self.current_block = saveCurrentBlock

        # build type of lambda
        assert (isinstance(block.ops[-1], Return))
        type = SSAValue.get(block.ops[-1].operands[0]).typ
        for arg in reversed(block.args):
            type = FunType.from_types(arg.typ, type)
        lambdaOp = Lambda.create([], [type], [], [],
                                 regions=[Region.from_block_list([block])])
        self._attach(lambdaOp)

        return lambdaOp

    def embed(self, *args: Union[Operation, SSAValue], resultType: Attribute,
              block: Block) -> Operation:
        # assert (len(block.args) == args.count)
        embedOp = Embed.create([self.getSSAValue(arg) for arg in args],
                               [resultType], [], [],
                               regions=[Region.from_block_list([block])])
        self._attach(embedOp)
        return embedOp

    def _return(self, value: Union[Operation, SSAValue]) -> Operation:
        returnOp = Return.create([value.results[0]])
        self._attach(returnOp)
        return returnOp

    # to do this properly the float additions in the open PR are required
    def literal(self, value: int, type: Attribute) -> Operation:
        literalOp = Literal.create(
            [], [f32], {"value": IntegerAttr.from_params(value, type)})
        self._attach(literalOp)
        return literalOp

    def lowering_unit(
            self, body: Callable[[BlockArgument, ...],
                                 List[Operation]]) -> Operation:
        flatten_list = lambda irregular_list: [
            element for item in irregular_list
            for element in Block.flatten_list(item)
        ] if type(irregular_list) is list else [irregular_list]

        self.current_block = Block.from_arg_types([])
        body()

        op = LoweringUnit.create(
            [], [], [], [],
            regions=[Region.from_block_list([self.current_block])])

        return op

    # def rise_dot_fused(self, left: ArrayType, right: ArrayType, arith: Arith):
    #     return self.reduce(
    #         self.literal(0, i32), self.zip(left, right), [
    #             self.tuple(self.scalar(f32), self.scalar(f32)),
    #             self.scalar(f32)
    #         ], lambda tuple, acc: [
    #             result := self.embed(
    #                 self.fst(tuple),
    #                 self.snd(tuple),
    #                 acc,
    #                 resultType=self.scalar(f32),
    #                 block=Block.from_callable(
    #                     [f32, f32, f32], lambda f, s, acc: [
    #                         product := arith.mulf(f, s),
    #                         result := arith.addf(product, acc),
    #                         stdReturn.get(result),
    #                     ])),
    #             self._return(result),
    #         ])
