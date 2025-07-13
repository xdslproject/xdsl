from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from xdsl.builder import Builder, InsertPoint
from xdsl.dialects.builtin import (
    FileLineColLoc,
    IntAttr,
    ModuleOp,
    StringAttr,
    TensorType,
    UnrankedTensorType,
    f64,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.utils.lexer import Location
from xdsl.utils.scoped_dict import ScopedDict

from ..dialects.toy import (
    AddOp,
    AnyTensorTypeF64,
    ConstantOp,
    FuncOp,
    FunctionType,
    GenericCallOp,
    MulOp,
    PrintOp,
    ReshapeOp,
    ReturnOp,
    TransposeOp,
    UnrankedTensorTypeF64,
)
from .toy_ast import (
    BinaryExprAST,
    CallExprAST,
    ExprAST,
    FunctionAST,
    LiteralExprAST,
    ModuleAST,
    NumberExprAST,
    PrintExprAST,
    PrototypeAST,
    ReturnExprAST,
    VarDeclExprAST,
    VariableExprAST,
)


class IRGenError(Exception):
    pass


@dataclass(init=False)
class IRGen:
    """
    Implementation of a simple MLIR emission from the Toy AST.

    This will emit operations that are specific to the Toy language, preserving
    the semantics of the language and (hopefully) allow to perform accurate
    analysis and transformation based on these high level semantics.
    """

    module: ModuleOp
    """A "module" matches a Toy source file: containing a list of functions."""

    builder: Builder
    """
    The builder is a helper class to create IR inside a function. The builder
    is stateful, in particular it keeps an "insertion point": this is where
    the next operations will be introduced."""

    symbol_table: ScopedDict[str, SSAValue] | None = None
    """
    The symbol table maps a variable name to a value in the current scope.
    Entering a function creates a new scope, and the function arguments are
    added to the mapping. When the processing of a function is terminated, the
    scope is destroyed and the mappings created in this scope are dropped."""

    def __init__(self):
        # We create an empty MLIR module and codegen functions one at a time and
        # add them to the module.
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        """
        Public API: convert the AST for a Toy module (source file) to an MLIR
        Module operation."""

        for f in module_ast.funcs:
            self.ir_gen_function(f)

        # Verify the module after we have finished constructing it, this will check
        # the structural properties of the IR and invoke any specific verifiers we
        # have on the Toy operations.
        try:
            self.module.verify()
        except Exception:
            print("module verification error")
            raise

        return self.module

    def loc(self, loc: Location):
        "Helper conversion for a Toy AST location to an MLIR location."
        return FileLineColLoc(StringAttr(loc.file), IntAttr(loc.line), IntAttr(loc.col))

    def declare(self, var: str, value: SSAValue) -> bool:
        """
        Declare a variable in the current scope, return success if the variable
        wasn't declared yet."""
        assert self.symbol_table is not None
        if var in self.symbol_table:
            return False
        self.symbol_table[var] = value
        return True

    def get_type(self, shape: list[int]) -> AnyTensorTypeF64:
        "Build a tensor type from a list of shape dimensions."
        # If the shape is empty, then this type is unranked.
        if len(shape):
            return TensorType(f64, shape)
        else:
            return UnrankedTensorTypeF64(f64)

    def ir_gen_proto(self, proto_ast: PrototypeAST) -> FuncOp:
        """
        Create the prototype for a function with as many arguments as the
        provided Toy AST prototype."""
        # location = self.loc(proto_ast.loc)

        # This is a generic function, the return type will be inferred later.
        # Arguments type are uniformly unranked tensors.
        func_type = FunctionType.from_lists(
            [self.get_type([])] * len(proto_ast.args), [self.get_type([])]
        )
        return self.builder.insert_op(FuncOp(proto_ast.name, func_type, Region()))

    def ir_gen_function(self, function_ast: FunctionAST) -> FuncOp:
        "Emit a new function and add it to the MLIR module."

        # keep builder for later
        parent_builder = self.builder

        # Create a scope in the symbol table to hold variable declarations.
        self.symbol_table = ScopedDict()

        proto_args = function_ast.proto.args

        # Create the block for the current function
        block = Block(
            arg_types=[UnrankedTensorType(f64) for _ in range(len(proto_args))]
        )
        self.builder = Builder(InsertPoint.at_end(block))

        # Declare all the function arguments in the symbol table.
        for name, value in zip(proto_args, block.args):
            self.declare(name.name, value)

        # Emit the body of the function.
        self.ir_gen_expr_list(function_ast.body)

        return_types = []

        # Implicitly return void if no return statement was emitted.
        return_op = None
        if block.ops:
            last_op = block.last_op
            if isinstance(last_op, ReturnOp):
                return_op = last_op
                if return_op.input is not None:
                    return_arg = return_op.input
                    return_types = [return_arg.type]
        if return_op is None:
            self.builder.insert_op(ReturnOp())

        input_types = [self.get_type([]) for _ in range(len(function_ast.proto.args))]

        func_type = FunctionType.from_lists(input_types, return_types)

        # main should be public, all the others private
        private = function_ast.proto.name != "main"

        # clean up
        self.symbol_table = None
        self.builder = parent_builder

        func = self.builder.insert_op(
            FuncOp(function_ast.proto.name, func_type, Region(block), private=private)
        )

        return func

    def ir_gen_binary_expr(self, binop: BinaryExprAST) -> SSAValue:
        "Emit a binary operation"

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

        lhs = self.ir_gen_expr(binop.lhs)
        rhs = self.ir_gen_expr(binop.rhs)

        # location = self.loc(binop.loc)

        # Derive the operation name from the binary operator. At the moment we only
        # support '+' and '*'.
        if binop.op == "+":
            op = self.builder.insert_op(AddOp(lhs, rhs))
        elif binop.op == "*":
            op = self.builder.insert_op(MulOp(lhs, rhs))
        else:
            raise IRGenError(f"Unsupported binary operation `{binop.op}`")

        return op.res

    def ir_gen_variable_expr(self, expr: VariableExprAST) -> SSAValue:
        """
        This is a reference to a variable in an expression. The variable is
        expected to have been declared and so should have a value in the symbol
        table, otherwise emit an error and return nullptr."""
        assert self.symbol_table is not None
        try:
            variable = self.symbol_table[expr.name]
            return variable
        except Exception as e:
            raise IRGenError(f"error: unknown variable `{expr.name}`") from e

    def ir_gen_return_expr(self, ret: ReturnExprAST):
        "Emit a return operation. This will return failure if any generation fails."

        # location = self.loc(binop.loc)

        # 'return' takes an optional expression, handle that case here.
        if ret.expr is not None:
            expr = self.ir_gen_expr(ret.expr)
        else:
            expr = None

        self.builder.insert_op(ReturnOp(expr))

    def ir_gen_literal_expr(self, lit: LiteralExprAST) -> SSAValue:
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

        # The attribute is a vector with a integer value per element
        # (number) in the array, see `collectData()` below for more details.
        data = self.collect_data(lit)

        # Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
        # method.
        op = self.builder.insert_op(ConstantOp.from_list(data, lit.dims))
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
            return [
                value for inner in expr.values for value in self.collect_data(inner)
            ]
        elif isinstance(expr, NumberExprAST):
            return [expr.val]
        else:
            raise IRGenError(
                f"Unsupported expr ({expr}) of type ({type(expr)}), "
                "expected literal or number expr"
            )

    def ir_gen_call_expr(self, call: CallExprAST) -> SSAValue:
        """
        Emit a call expression. It emits specific operations for the `transpose`
        builtin. Other identifiers are assumed to be user-defined functions.
        """
        assert self.symbol_table is not None
        callee = call.callee

        #    auto location = loc(call.loc());
        # Codegen the operands first.
        operands = [self.ir_gen_expr(expr) for expr in call.args]

        # Builtin calls have their custom operation, meaning this is a
        # straightforward emission.
        if callee == "transpose":
            if len(operands) != 1:
                raise IRGenError(
                    "MLIR codegen encountered an error: toy.transpose "
                    "does not accept multiple arguments"
                )
            op = self.builder.insert_op(TransposeOp(operands[0]))
            return op.res

        # Otherwise this is a call to a user-defined function. Calls to
        # user-defined functions are mapped to a custom call that takes the callee
        # name as an attribute.
        op = self.builder.insert_op(
            GenericCallOp(callee, operands, [UnrankedTensorTypeF64(f64)])
        )

        return op.res[0]

    def ir_gen_print_expr(self, call: PrintExprAST):
        """
        Emit a print expression. It emits specific operations for two builtins:
        transpose(x) and print(x).
        """
        arg = self.ir_gen_expr(call.arg)
        self.builder.insert_op(PrintOp(arg))

    def ir_gen_number_expr(self, num: NumberExprAST) -> SSAValue:
        "Emit a constant for a single number"
        constant_op = self.builder.insert_op(ConstantOp.from_value(num.val))
        return constant_op.res

    def ir_gen_expr(self, expr: ExprAST) -> SSAValue:
        "Dispatch codegen for the right expression subclass using RTTI."

        if isinstance(expr, BinaryExprAST):
            return self.ir_gen_binary_expr(expr)
        if isinstance(expr, VariableExprAST):
            return self.ir_gen_variable_expr(expr)
        if isinstance(expr, LiteralExprAST):
            return self.ir_gen_literal_expr(expr)
        if isinstance(expr, CallExprAST):
            return self.ir_gen_call_expr(expr)
        if isinstance(expr, NumberExprAST):
            return self.ir_gen_number_expr(expr)
        else:
            raise IRGenError(
                f"MLIR codegen encountered an unhandled expr kind '{type(expr).__name__}'"
            )

    def ir_gen_var_decl_expr(self, vardecl: VarDeclExprAST) -> SSAValue:
        """
        Handle a variable declaration, we'll codegen the expression that forms the
        initializer and record the value in the symbol table before returning it.
        Future expressions will be able to reference this variable through symbol
        table lookup.
        """

        value = self.ir_gen_expr(vardecl.expr)

        # We have the initializer value, but in case the variable was declared
        # with specific shape, we emit a "reshape" operation. It will get
        # optimized out later as needed.
        if len(vardecl.varType.shape):
            reshape_op = self.builder.insert_op(ReshapeOp(value, vardecl.varType.shape))

            value = reshape_op.res

        # Register the value in the symbol table.
        self.declare(vardecl.name, value)

        return value

    def ir_gen_expr_list(self, exprs: Iterable[ExprAST]) -> None:
        "Codegen a list of expressions, raise error if one of them hit an error."
        assert self.symbol_table is not None

        for expr in exprs:
            # Specific handling for variable declarations, return statement, and
            # print. These can only appear in block list and not in nested
            # expressions.
            if isinstance(expr, VarDeclExprAST):
                self.ir_gen_var_decl_expr(expr)
            elif isinstance(expr, ReturnExprAST):
                self.ir_gen_return_expr(expr)
            elif isinstance(expr, PrintExprAST):
                self.ir_gen_print_expr(expr)
            else:
                # Generic expression dispatch codegen.
                self.ir_gen_expr(expr)
