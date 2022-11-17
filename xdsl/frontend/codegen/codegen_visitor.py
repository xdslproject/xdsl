import ast

from dataclasses import dataclass, field
from typing import Any, Dict, List
from xdsl.frontend.codegen.resolver import OpResolver
from xdsl.frontend.codegen.type_manager import TypeManager
from xdsl.frontend.codegen.utils.codegen_for import check_for_loop_valid, codegen_affine_for_loop, codegen_scf_for_loop, is_affine_for_loop

from xdsl.frontend.codegen.utils.codegen_function import check_function_signature, get_argument_types, get_return_types
from xdsl.dialects import builtin, func, scf, symref, arith, affine, unimplemented
from xdsl.frontend.codegen.exception import CodegenException, prettify
from xdsl.frontend.codegen.inserter import OpInserter
from xdsl.frontend.codegen.type_conversion import TypeHintConverter
from xdsl.ir import Attribute, Operation, Block, Region, SSAValue


@dataclass
class CodegenVisitor(ast.NodeVisitor):
    """Visitor that generates xDSL from the frontend AST."""

    globals: Dict[str, Any]
    """Imports and other global information from the module usefule for looking up classes."""

    type_manager: TypeManager
    """Casts types."""

    hint_converter: TypeHintConverter = field(init=False)
    """Class responsible for type hint conversion to xDSL types."""

    # TODO: fix symbol table to allow cross-module scoping.
    symbol_table: Dict[str, Attribute] = field(init=False)
    """Symbol table used to query types for symref symbols per function."""

    # TODO: fix function table to allow cross-module scoping.
    # TODO: change this design to allow arbitrary function orderings per 
    # module.
    function_table: Dict[str, func.FuncOp] = field(init=False)
    """Function table used to query built functions per module."""

    inserter: OpInserter = field(init=False)
    """Class responsible for inserting new xDSL ops at the right place."""

    def __init__(self, globals: Dict[str, Any]):
        inserter = OpInserter()
        
        self.globals = globals
        self.type_manager = TypeManager(inserter)
        self.hint_converter = TypeHintConverter(globals)
        self.symbol_table = dict()
        self.function_table = dict()
        self.inserter = inserter

    def visit(self, node: ast.AST):
        return super().visit(node)

    def generic_visit(self, node: ast.AST):
        raise CodegenException(f"visitor for node {node} does not exist")

    def visit_Expr(self, node: ast.Expr):
        self.visit(node.value)

    def _cast(self, dst_ty: Attribute, value_ty: Attribute, value: SSAValue | Operation):
        # There are a lot of different casts, for now just put a placeholder op instead
        # of arith.trunci and friends.
        # TODO: implement casts.
        self.inserter.insert_op(unimplemented.Cast.get(value, dst_ty))
        return self.inserter.get_operand()

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """
        Visits type-annotated assignment operation, e.g.

        a: i32 = 3
        """
        # First, find the type of the LHS based on the type hint and create a new
        # symref declaration.
        lhs_ty = self.hint_converter.convert_hint(node.annotation)
        declare_op = symref.Declare.get(node.target.id)
        self.inserter.insert_op(declare_op)

        # Make sure the symbol table knows the type information. For now we only allow
        # referring to the symbols within the function, but in future that should change!
        # TODO: fix symbol table.
        self.symbol_table[node.target.id] = lhs_ty

        # Also, smake sure that we know what type the RHS expression should have.
        # self.state.inferred_type = lhs_ty

        # Next visit RHS and get the value of that expression and its type.
        self.visit(node.value)
        rhs = self.inserter.get_operand()

        # Now, it can be that RHS already used the LHS type, e.g. when visiting
        # a: i32 = 0, constant visitor used the type inferred from the type hint
        # to create 0 constant. ALternatively, it can happen that the type of LHS was
        # not used! For example, if we have x: i32 = 0; y: i64 = x, the type of x
        # is i32 instead of i64, so we must do the type cobersion.
    
        # TODO: check types and add implicit casts if necessary.
        rhs = self.type_manager.match(lhs_ty, rhs)

        update_op = symref.Update.get(node.target.id, rhs)
        self.inserter.insert_op(update_op)
        # self.state.inferred_type = None

    def visit_Assign(self, node: ast.Assign):
        """
        Visits assignment operation, e.g.

        a = 3
        """
        # Get the rhs first.
        self.visit(node.value)
        rhs = self.inserter.get_operand()

        if isinstance(node.targets[0], ast.Name):
            lhs_ty = self.symbol_table[node.targets[0].id]
            # self.state.inferred_type = lhs_ty

            # TODO: check types and add implicit casts if necessary.
            rhs = self.type_manager.match(lhs_ty, rhs)

            update_op = symref.Update.get(node.targets[0].id, rhs)
            self.inserter.insert_op(update_op)
            return
        
        if isinstance(node.targets[0], ast.Subscript):
            node = node.targets[0]
            indices: List[SSAValue] = []
            while isinstance(node, ast.Subscript):
                self.visit(node.slice)
                index = self.inserter.get_operand()
                indices.append(index)
                node = node.value
        
            indices = list(reversed(indices))
            self.visit(node)
            indexed_value = self.inserter.get_operand()

            frontend_type = self.hint_converter.type_backward_map[indexed_value.typ.__class__]

            # TODO: check types and add implicit casts if necessary.
            # we have to check the element type here.

            resolver = OpResolver.resolve_op_overload("__setitem__", frontend_type)
            if resolver is None:
                raise CodegenException("operator __setitem__() is not supported")

            op = resolver()(rhs, indexed_value, *indices)
            self.inserter.insert_op(op)


    def visit_BinOp(self, node: ast.BinOp):
        """
        Visits a binary operation.
        """
        self.visit(node.right)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()

        # TODO: check types and add implicit casts if necessary.
        rhs = self.type_manager.match(lhs.typ, rhs)

        # Try to resolve thi sbinary operator.
        op_name: str = node.op.__class__.__name__
        frontend_type = self.hint_converter.type_backward_map[lhs.typ.__class__]
        resolver = OpResolver.resolve_op_overload(op_name, frontend_type)
        if resolver is None:
            raise CodegenException(f"binary operator {op_name} is not supported")
        
        # If resolved, we should get a binary op.
        op = resolver()(lhs, rhs)
        self.inserter.insert_op(op)

    def visit_Compare(self, node: ast.Compare):
        """
        Visits a comparison operation.
        """
        # First, allow a single comparison only.
        if len(node.comparators) != 1 or len(node.ops) != 1:
            raise CodegenException(f"require a single comparator and op, found {len(node.comparators)} and {len(node.ops)}")

        op = node.ops[0]
        comp = node.comparators[0]

        # Get the values we compare.
        self.visit(comp)
        rhs = self.inserter.get_operand()
        self.visit(node.left)
        lhs = self.inserter.get_operand()

        # TODO: chech types and add implicit casts if necessary.
        rhs = self.type_manager.match(lhs.typ, rhs)

        op_name: str = op.__class__.__name__
        frontend_type = self.hint_converter.type_backward_map[lhs.typ.__class__]
        resolver = OpResolver.resolve_op_overload(op_name, frontend_type)
        if resolver is None:
            raise CodegenException(f"comparison operator {op_name} is not supported")

        # Map from comparison operation to mnemonics.
        # TODO: what about unsigned mnemonics like 'ugt'?
        cmp_op_to_mnemonic = {
            "Eq": "eq",
            "NotEq": "ne",
            "LtE": "sle",
            "Lt": "slt",
            "Gt": "sgt",
            "GtE": "sge",
        }

        if op_name not in cmp_op_to_mnemonic:
            raise CodegenException(f"comparison operator {op_name} is not supported")
        
        # If resolved, we should get a binary op.
        op = resolver()(lhs, rhs, cmp_op_to_mnemonic[op_name])
        self.inserter.insert_op(op)

    def visit_Call(self, node: ast.Call):
        """
        Visits a function call, which can be a library function, dialect
        operation or user defined function.
        """

        # Process the arguments first.
        num_args = len(node.args)
        for arg in reversed(node.args):
            self.visit(arg)
        
        # Collect the operand list and the type information.
        # TODO: here we allow operands to be python types, which seems not that great. maybe
        # we should fix this later (al because of strings!)
        operand_types: List[Attribute] = []
        operands: List[SSAValue] = []
        for i in range(num_args):
            operand = self.inserter.get_operand()
            operands.append(operand)
            if isinstance(operand, str):
                operand_types.append(type(operand))
            else:
                operand_types.append(operand.typ)

        # Call can be made to a user-defined function, process this case first.
        if isinstance(node.func, ast.Name) and node.func.id in self.function_table:
            callee = self.function_table[node.func.id]
            callee_operand_types = callee.function_type.inputs.data

            # Check operand types.
            for i in range(num_args):
                actual_ty = operand_types[i]
                expected_ty = callee_operand_types[i]
                if actual_ty != expected_ty:
                    # TODO: implicit cast
                    operands[i] = self.type_manager.match(expected_ty, operands[i])
                    # raise CodegenException(f"wrong argument type at position {i} when calling '{node.func.id}', expected {prettify(expected_ty)}, got {prettify(actual_ty)}")

            # Operand types match, so we can create a call operation and insert
            # it in the current block.
            call_op = func.Call.get(node.func.id, operands, callee.function_type.outputs.data) 
            self.inserter.insert_op(call_op)
            return

        # Otherwise, get the module and the function names.
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            module_name = self.globals[func_name].__module__
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = node.func.attr
            module_name = self.globals[node.func.value.id].__name__
        else:
            raise CodegenException("function calls are supported only in the form M.F() or F()")

        resolver = OpResolver.resolve_method(module_name, func_name)
        if resolver is None:
            # TODO: what about other standard libraries, max, sum, etc? They should be
            # handled here.
            raise CodegenException(f"unknown function {func_name} from {module_name}")
        
        op = resolver()(*operands)
        self.inserter.insert_op(op)

    def visit_Constant(self, node: ast.Constant):
        """
        Visits a constant value.
        """
        if isinstance(node.value, str):
            self.inserter.stack.append(node.value)
        elif isinstance(node.value, int):
            target_ty = self.type_manager.default_type(type(node.value))
            value_attr = builtin.IntegerAttr.from_int_and_width(node.value, target_ty.width.data)
            constant_op = arith.Constant.from_attr(value_attr, target_ty)
            self.inserter.insert_op(constant_op)
        elif isinstance(node.value, float):
            target_ty = self.type_manager.default_type(type(node.value))
            value_attr = builtin.FloatAttr.from_float_and_width(node.value, target_ty.width.data)
            constant_op = arith.Constant.from_attr(value_attr, target_ty)
            self.inserter.insert_op(constant_op)
        else:
            raise CodegenException(f"unknown constant {node.value} of type {type(node.value)}")

    def visit_For(self, node: ast.For):
        """Visits a for loop and creates scf.for or affine.for operation."""

        # First, check if this loop can be lowered to xDSL.
        check_for_loop_valid(node)

        # Next, we have to check if the loop is affine: for now we simply
        # check if all range arguments are constants. If not, we have to generate scf.for
        if is_affine_for_loop(node):
            codegen_affine_for_loop(self.inserter, node, self.visit)
        else:
            codegen_scf_for_loop(self.inserter, node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visits a function definition, e.g.

        def foo():
            ...
        """
        # First, check if function signature is valid.
        check_function_signature(node)

        # Then, convert type in the function signature.
        arg_types = get_argument_types(node, self.hint_converter)
        return_types = get_return_types(node, self.hint_converter)

        # Create a region for the function body and entry block.
        entry_block = Block()
        body_region = Region.from_block_list([entry_block])
        func_op = func.FuncOp.from_region(node.name, arg_types, return_types, body_region)
        self.inserter.insert_op(func_op)
        self.inserter.set_insertion_point_from_block(entry_block)

        # Reset the symbol table.
        # TODO: this doesn't handle global variables at the moment.
        self.symbol_table = dict()

        # All arguments are declared using symref.
        for i, arg in enumerate(node.args.args):
            symbol_name = arg.arg
            arg = entry_block.insert_arg(arg_types[i], i)
            entry_block.add_op(symref.Declare.get(symbol_name))
            self.symbol_table[symbol_name] = arg_types[i]
            entry_block.add_op(symref.Update.get(symbol_name, arg))

        # Parse function body.
        for stmt in node.body:
            self.visit(stmt)

        # Check that return statement has been inserted. If not, do that. This way we
        # handle cases like:
        #
        # def foo():
        #   # some code without 'return' at the end
        #
        ops = self.inserter.ip.ops
        if len(ops) == 0 or not isinstance(ops[-1], func.Return):
            return_types = func_op.function_type.outputs.data
            assert len(return_types) <= 1

            if len(return_types) != 0:
                func_name = func_op.attributes["sym_name"].data
                raise CodegenException(f"expected 1 return type, got 0 in function {func_name}")
            self.inserter.insert_op(func.Return.get())

        # Move on with code generation for the next operation and record
        # the function to allow calls to it.
        self.function_table[func_op.attributes["sym_name"].data] = func_op
        self.inserter.set_insertion_point_from_op(func_op.parent_op())

    def visit_If(self, node: ast.If):
        # Get the condition.
        self.visit(node.test)
        cond = self.inserter.get_operand()
        prev_insertion_point = self.inserter.ip

        # Process true region
        true_region = Region.from_block_list([Block()])
        self.inserter.set_insertion_point_from_region(true_region)
        for stmt in node.body:
            self.visit(stmt)

        # Process false region
        false_region = Region.from_block_list([Block()])
        self.inserter.set_insertion_point_from_region(false_region)
        for stmt in node.orelse:
            self.visit(stmt)
        
        # Add yield operations.
        true_region.blocks[-1].add_op(scf.Yield.get())
        false_region.blocks[-1].add_op(scf.Yield.get())

        # Reset insertion point to add scf.if
        self.inserter.set_insertion_point_from_block(prev_insertion_point)
        op = scf.If.get(cond, [], true_region, false_region)
        self.inserter.insert_op(op)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        # Get the condition.
        self.visit(node.test)
        cond = self.inserter.get_operand()
        prev_insertion_point = self.inserter.ip

        # Process true region
        true_region = Region.from_block_list([Block()])
        self.inserter.set_insertion_point_from_region(true_region)
        self.visit(node.body)
        true_result = self.inserter.get_operand()
        self.inserter.insert_op(scf.Yield.get(true_result))

        # Process false region
        false_region = Region.from_block_list([Block()])
        self.inserter.set_insertion_point_from_region(false_region)
        self.visit(node.orelse)
        false_result = self.inserter.get_operand()
        self.inserter.insert_op(scf.Yield.get(false_result))

        # TODO: check types are the same or add an implicit cast.

        # Reset insertion point to add scf.if
        self.inserter.set_insertion_point_from_block(prev_insertion_point)
        op = scf.If.get(cond, [true_result.typ], true_region, false_region)
        self.inserter.insert_op(op)
        self.inserter.set_insertion_point_from_op(op.parent_op())

    def visit_Name(self, node: ast.Name):
        """
        Visits a named variable - it is a stack-allocated variable or an argument.
        """

        # TODO: this assumes variable is local and no global variables exist. When
        # the sysmbol table is changed so that we can use globals, this can change.
        ty = self.symbol_table[node.id]
        fetch_op = symref.Fetch.get(node.id, ty)
        self.inserter.insert_op(fetch_op)

    def visit_Return(self, node: ast.Return):
        """
        Visits a return statement in the function.
        """

        # First of all, we should only be able to return if the statement is directly
        # in the function. Cases like:
        #
        # def foo(cond: i1):
        #   if cond:
        #     return 1
        #   else:
        #     return 0
        #
        # are not allowed at the moment.
        # TODO: support cases like mentioned above with multiple terminators.
        parent_op = self.inserter.ip.parent_op()
        if not isinstance(self.inserter.ip.parent_op(), func.FuncOp):
            raise CodegenException("return statement should be placed only at the end of the function body")

        func_name = parent_op.attributes["sym_name"].data
        func_return_types = parent_op.function_type.outputs.data

        if node.value is None:
            # Return nothing, check function signature matches.
            if len(func_return_types) != 0:
                raise CodegenException(f"expected non-zero return types, got 0 in function '{func_name}'")
            return_op = func.Return.get()
            self.inserter.insert_op(return_op)
        else:
            # Return some type, check function signature matches as well.
            self.visit(node.value)
            operand = self.inserter.get_operand()

            # TODO: this can be dropped when function is allowed to return more
            # than one type.
            if len(func_return_types) > 1:
                raise CodegenException(f"expected less than 2 return types, got {len(func_return_types)} in function '{func_name}'")

            if len(func_return_types) == 0:
                raise CodegenException(f"expected 0 return types, got 1 in function {func_name}")
            if func_return_types[0] != operand.typ:
                # TODO: implicit cast
                operand = self.type_manager.match(func_return_types[0], operand)
                # raise CodegenException(f"expected {prettify(func_return_types[0])} return type, got {prettify(operand.typ)} in function {func_name}")

            return_op = func.Return.get(operand)
            self.inserter.insert_op(return_op)

    def visit_With(self, node: ast.With):
        """
        Visits a with block which represents a new module.
        """
        # In the future, with can also be used for regions. But let's
        # not worry about that at the moment.
        # TODO: support with Region():
        module_op = builtin.ModuleOp.from_region_or_ops([])

        # TODO: we have per module function table. Instead it would be
        # nice to call functions from other modules.
        self.function_table = dict()

        # Proceed with visitng the module.
        self.inserter.insert_op(module_op)
        self.inserter.set_insertion_point_from_op(module_op)
        for stmt in node.body:
            self.visit(stmt)

        # Reset insertion points back.
        self.inserter.set_insertion_point_from_op(module_op.parent_op())

    def visit_Pass(self, node: ast.Pass):
        # Special case: function can return nothing and be implemented using pass, e.g.
        #  def foo():
        #    pass
        # Therefore, we have to explicitly add func.return unless type signature
        # says otherwise.
        parent_op = self.inserter.ip.parent_op()
        if isinstance(parent_op, func.FuncOp):
            func_name = parent_op.attributes["sym_name"].data
            return_types = parent_op.function_type.outputs.data

            # Technically this is not fully correct because 'pass' in Python just
            # means "do nothing". So we should only produce a function declaration
            # instead.
            # TODO: fix this.
            if len(return_types) != 0:
                raise CodegenException(f"expected non-zero return types, got 0 in function '{func_name}'")

            self.inserter.insert_op(func.Return.get())

    def visit_Subscript(self, node: ast.Subscript):
        """
        Visits subscript expressions like x[i][j][k]
        """
        indices: List[SSAValue] = []
        while isinstance(node, ast.Subscript):
            self.visit(node.slice)
            index = self.inserter.get_operand()
            indices.append(index)
            node = node.value
        
        indices = list(reversed(indices))
        self.visit(node)
        indexed_value = self.inserter.get_operand()

        frontend_type = self.hint_converter.type_backward_map[indexed_value.typ.__class__]

        resolver = OpResolver.resolve_op_overload("__getitem__", frontend_type)
        if resolver is None:
            raise CodegenException("operator __getitem__() is not supported")

        op = resolver()(indexed_value, *indices)
        self.inserter.insert_op(op)
