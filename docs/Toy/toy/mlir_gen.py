from __future__ import annotations

from dataclasses import field, dataclass
from typing import Iterable

from xdsl.ir import MLContext, SSAValue, Operation, Block, Region
from xdsl.dialects.builtin import ModuleOp, f64, TensorType, UnrankedTensorType

from toy.location import Location
from toy.toy_ast import (LiteralExprAST, ModuleAST, NumberExprAST, PrototypeAST, VariableExprAST, VarDeclExprAST, ReturnExprAST, PrintExprAST, FunctionAST, ExprAST, CallExprAST, BinaryExprAST)



from .dialect import (TensorTypeF64, UnrankedTensorTypeF64, AddOp, MulOp, FuncOp, FunctionType, ReturnOp, ConstantOp, GenericCallOp, TransposeOp, ReshapeOp, PrintOp)

class MLIRGenError(Exception):
    pass

@dataclass
class OpBuilder:
    ctx: MLContext

@dataclass
class ScopedSymbolTable:
    'A mapping from variable names to SSAValues, append-only'
    table: dict[str, SSAValue] = field(default_factory=dict)


    def __contains__(self, __o: object) -> bool:
        return __o in self.table

    def __getitem__(self, __key: str) -> SSAValue:
        return self.table[__key]

    def __setitem__(self, __key: str, __value: SSAValue) -> None:
        if __key in self:
            raise AssertionError(f'Cannot add value for key {__key} in scope {self}')
        self.table[__key] = __value


class MLIRGen:
    """
    Implementation of a simple MLIR emission from the Toy AST.

    This will emit operations that are specific to the Toy language, preserving
    the semantics of the language and (hopefully) allow to perform accurate
    analysis and transformation based on these high level semantics.
    """

    # module: ModuleOp | None = None
    # 'A "module" matches a Toy source file: containing a list of functions.'

    # builder: OpBuilder
    # """
    # The builder is a helper class to create IR inside a function. The builder
    # is stateful, in particular it keeps an "insertion point": this is where
    # the next operations will be introduced."""

    block: Block | None = None
    symbol_table: ScopedSymbolTable | None = None

    """
    The symbol table maps a variable name to a value in the current scope.
    Entering a function creates a new scope, and the function arguments are
    added to the mapping. When the processing of a function is terminated, the
    scope is destroyed and the mappings created in this scope are dropped."""

    def __init__(self, ctx: MLContext):
        self.builder = OpBuilder(ctx)

    def mlir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        """
        Public API: convert the AST for a Toy module (source file) to an MLIR
        Module operation."""
        
        # We create an empty MLIR module and codegen functions one at a time and
        # add them to the module.
        # self.module = ModuleOp.create(regions=[Region()])

        functions: list[Operation] = []

        for f in module_ast.funcs:
            functions.append(self.mlir_gen_function(f))

        module = ModuleOp.from_region_or_ops(functions)

        # Verify the module after we have finished constructing it, this will check
        # the structural properties of the IR and invoke any specific verifiers we
        # have on the Toy operations.
        try:
            module.verify()
        except Exception:
            print('module verification error')
            raise

        return module

    def loc(self, loc: Location):
        'Helper conversion for a Toy AST location to an MLIR location.'
        # TODO: Need location support in xDSL
        # return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line, loc.col);
        pass
    
    def declare(self, var: str, value: SSAValue) -> bool:
        """
        Declare a variable in the current scope, return success if the variable
        wasn't declared yet."""
        assert self.symbol_table is not None
        if var in self.symbol_table:
            return False
        self.symbol_table[var] = value
        return True

    def get_type(self, shape: list[int]) -> TensorTypeF64 | UnrankedTensorTypeF64:
        'Build a tensor type from a list of shape dimensions.'
        # If the shape is empty, then this type is unranked.
        if len(shape):
            return TensorType.from_type_and_list(f64, shape)
        else:
            return UnrankedTensorTypeF64.from_type(f64)

    def mlir_gen_proto(self, proto_ast: PrototypeAST) -> FuncOp:
        """
        Create the prototype for an MLIR function with as many arguments as the
        provided Toy AST prototype."""
        # location = self.loc(proto_ast.loc)

        # This is a generic function, the return type will be inferred later.
        # Arguments type are uniformly unranked tensors.
        func_type = FunctionType.from_lists([self.get_type([])] * len(proto_ast.args), [self.get_type([])])
        return FuncOp.from_region(proto_ast.name, func_type, Region())

    def mlir_gen_function(self, function_ast: FunctionAST) -> FuncOp:
        'Emit a new function and add it to the MLIR module.'

        # Create a scope in the symbol table to hold variable declarations.
        self.symbol_table = ScopedSymbolTable()
        
        proto_args = function_ast.proto.args
        
        # Create the MLIR block for the current function
        self.block = Block.from_arg_types([UnrankedTensorType.from_type(f64) for _ in range(len(proto_args))])

        # Declare all the function arguments in the symbol table.
        for name, value in zip(proto_args, self.block.args):
            self.declare(name.name, value)
        
        # Emit the body of the function.
        self.mlir_gen_expr_list(function_ast.body)

        return_types = []

        # Implicitly return void if no return statement was emitted.
        return_op = None
        if len(self.block.ops):
            last_op = self.block.ops[-1]
            if isinstance(last_op, ReturnOp):
                return_op = last_op
                if return_op.input is not None:
                    return_arg = return_op.input
                    return_types = [return_arg.typ]
        if return_op is None:
            return_op = ReturnOp.from_input()
            self.block.add_op(return_op)
            

        input_types = [self.get_type([]) for _ in range(len(function_ast.proto.args))]

        func_type = FunctionType.from_lists(input_types, return_types)

        # main should be public, all the others private
        private = function_ast.proto.name != 'main'

        func = FuncOp.from_region(function_ast.proto.name, func_type, 
                                  Region.from_block_list([self.block]), private=private)

        # clean up
        self.symbol_table = None
        self.block = None

        return func


    def mlir_gen_binary_expr(self, binop: BinaryExprAST) -> SSAValue:
        'Emit a binary operation'
        assert self.block is not None

        # First emit the operations for each side of the operation before emitting
        # the operation itself. For example if the expression is `a + foo(a)`
        # 1) First it will visiting the LHS, which will return a reference to the
        #    value holding `a`. This value should have been emitted at declaration
        #    time and registered in the symbol table, so nothing would be
        #    codegen'd. If the value is not in the symbol table, an error has been
        #    emitted and nullptr is returned.
        # 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
        #    and the result value is returned. If an error occurs we get a nullptr
        #    and propagate.

        lhs = self.mlir_gen_expr(binop.lhs)
        rhs = self.mlir_gen_expr(binop.rhs)

        # location = self.loc(binop.loc)
        
        # Derive the operation name from the binary operator. At the moment we only
        # support '+' and '*'.
        match binop.op:
            case '+':
                op = AddOp.from_summands(lhs, rhs)
            case '*':
                op = MulOp.from_summands(lhs, rhs)
            case _:
                self.error(f'Unsupported binary operation `{binop.op}`')
        
        self.block.add_op(op)

        return op.res

    def mlir_gen_variable_expr(self, expr: VariableExprAST) -> SSAValue:
        """
        This is a reference to a variable in an expression. The variable is
        expected to have been declared and so should have a value in the symbol
        table, otherwise emit an error and return nullptr."""
        assert self.symbol_table is not None
        try:
            variable = self.symbol_table[expr.name]
            return variable
        except Exception as e:
            self.error(f'error: unknown variable `{expr.name}`', e)

    def mlir_gen_return_expr(self, ret: ReturnExprAST):
        'Emit a return operation. This will return failure if any generation fails.'
        assert self.block is not None

        # location = self.loc(binop.loc)

        # 'return' takes an optional expression, handle that case here.
        if ret.expr is not None:
            expr = self.mlir_gen_expr(ret.expr)
        else:
            expr = None

        return_op = ReturnOp.from_input(expr)
        self.block.add_op(return_op)

    def mlir_gen_literal_expr(self, lit: LiteralExprAST) -> SSAValue:
        """
        Emit a literal/constant array. It will be emitted as a flattened array of
        data in an Attribute attached to a `toy.constant` operation.
        See documentation on [Attributes](LangRef.md#attributes) for more details.
        Here is an excerpt:
        Attributes are the mechanism for specifying constant data in MLIR in
        places where a variable is never allowed [...]. They consist of a name
        and a concrete attribute value. The set of expected attributes, their
        structure, and their interpretation are all contextually dependent on
        what they are attached to.
        Example, the source level statement:
        var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
        will be converted to:
        %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
            [[1.000000e+00, 2.000000e+00, 3.000000e+00],
            [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
        """
        assert self.block is not None

        # The attribute is a vector with a integer value per element
        # (number) in the array, see `collectData()` below for more details.
        data = self.collect_data(lit)

        # Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
        # method.
        op = ConstantOp.from_list(data, lit.dims)
        self.block.add_op(op)
        return op.res

    def collect_data(self, expr: ExprAST) -> list[float]:
        """
        Helper function to accumulate the data that compose an array
        literal. It flattens the nested structure in the supplied vector. For
        example with this array:
         [[1, 2], [3, 4]]
        we will generate:
         [ 1, 2, 3, 4 ]
        Individual numbers are represented as doubles.
        Attributes are the way MLIR attaches constant to operations.
        """

        if isinstance(expr, LiteralExprAST):
            return expr.flattened_values()
        elif isinstance(expr, NumberExprAST):
            return [expr.val]
        else:
            self.error(
                f'Unsupported expr ({expr}) of type ({type(expr)}), '
                'expected literal or number expr')


    def mlir_gen_call_expr(self, call: CallExprAST) -> SSAValue:
        """
        Emit a call expression. It emits specific operations for the `transpose`
        builtin. Other identifiers are assumed to be user-defined functions.
        """
        assert self.block is not None
        assert self.symbol_table is not None
        callee = call.callee

        #    auto location = loc(call.loc());
        # Codegen the operands first.
        operands = [self.mlir_gen_expr(expr) for expr in call.args]

        # Builtin calls have their custom operation, meaning this is a
        # straightforward emission.
        if callee == 'transpose':
            if len(operands) != 1:
                self.error(
                    "MLIR codegen encountered an error: toy.transpose "
                    "does not accept multiple arguments")
            op = TransposeOp.from_input(operands[0])
            self.block.add_op(op)
            return op.res
        
        # Otherwise this is a call to a user-defined function. Calls to
        # user-defined functions are mapped to a custom call that takes the callee
        # name as an attribute.
        op = GenericCallOp.get(callee, operands, [UnrankedTensorTypeF64.from_type(f64)])
        self.block.add_op(op)
        return op.res[0]

    def mlir_gen_print_expr(self, call: PrintExprAST):
        """
        Emit a print expression. It emits specific operations for two builtins:
        transpose(x) and print(x).
        """
        assert self.block is not None
        arg = self.mlir_gen_expr(call.arg)
        op = PrintOp.from_input(arg)
        self.block.add_op(op)


    def mlir_gen_number_expr(self, num: NumberExprAST) -> SSAValue:
        'Emit a constant for a single number'
        #  mlir::Value mlirGen(NumberExprAST &num) {
        #    return builder.create<ConstantOp>(loc(num.loc()), num.getValue());
        #  }
        assert self.block is not None

        constant_op = ConstantOp.from_list([num.val], [])
        self.block.add_op(constant_op)
        return constant_op.res


    def mlir_gen_expr(self, expr: ExprAST) -> SSAValue:
        'Dispatch codegen for the right expression subclass using RTTI.'

        match expr:
            case BinaryExprAST(): return self.mlir_gen_binary_expr(expr)
            case VariableExprAST(): return self.mlir_gen_variable_expr(expr)
            case LiteralExprAST(): return self.mlir_gen_literal_expr(expr)
            case CallExprAST(): return self.mlir_gen_call_expr(expr)
            case NumberExprAST(): return self.mlir_gen_number_expr(expr)
            case _:
                self.error(
                    f"MLIR codegen encountered an unhandled expr kind '{expr.kind}'"
                )


    def mlir_gen_var_decl_expr(self, vardecl: VarDeclExprAST) -> SSAValue:
        """
        Handle a variable declaration, we'll codegen the expression that forms the
        initializer and record the value in the symbol table before returning it.
        Future expressions will be able to reference this variable through symbol
        table lookup.
        """
        assert self.block is not None

        value = self.mlir_gen_expr(vardecl.expr)

        # We have the initializer value, but in case the variable was declared
        # with specific shape, we emit a "reshape" operation. It will get
        # optimized out later as needed.
        if len(vardecl.varType.shape):
            reshape_op = ReshapeOp.from_input(value, vardecl.varType.shape)
            self.block.add_op(reshape_op)
            value = reshape_op.res

        # Register the value in the symbol table.
        self.declare(vardecl.name, value)

        return value

    def mlir_gen_expr_list(self, exprs: Iterable[ExprAST]) -> None:
        'Codegen a list of expressions, raise error if one of them hit an error.'
        assert self.symbol_table is not None

        for expr in exprs:
            # Specific handling for variable declarations, return statement, and
            # print. These can only appear in block list and not in nested
            # expressions.
            match expr:
                case VarDeclExprAST(): self.mlir_gen_var_decl_expr(expr)
                case ReturnExprAST(): self.mlir_gen_return_expr(expr)
                case PrintExprAST(): self.mlir_gen_print_expr(expr)
                # Generic expression dispatch codegen.
                case _: self.mlir_gen_expr(expr)

    def error(self, message: str, cause: Exception | None = None):
        raise MLIRGenError(message) from cause
