import ast

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from xdsl.frontend.codegen.functions import FunctionInfo
from xdsl.frontend.codegen.resolver import OpResolver
from xdsl.frontend.codegen.type_inference import TypeInference
from xdsl.frontend.codegen.type_manager import TypeManager
from xdsl.dialects import builtin, cf, func, scf, symref, arith, affine, tensor, unimplemented
from xdsl.frontend.codegen.exception import CodegenInternalException
from xdsl.frontend.codegen.inserter import OpInserter
from xdsl.frontend.codegen.type_conversion import TypeConverter
from xdsl.ir import Attribute, Operation, Block, Region, SSAValue


@dataclass
class CodegenVisitor(ast.NodeVisitor):
    """Visitor that generates xDSL from the frontend AST."""

    globals: Dict[str, Any]
    """Imports and other global information from the module useful for looking up classes, etc."""

    type_manager: TypeManager = field(init=False)
    """Casts types."""

    induction_vars: Dict[str, SSAValue] = field(init=False)

    # TODO: fix symbol table to allow cross-module scoping.
    symbol_table: Dict[str, Attribute] = field(init=False)
    symbol_idx: Dict[str, int] = field(init=False)
    """Symbol table used to query types for symref symbols per function."""

    inserter: OpInserter = field(init=False)
    """Class responsible for inserting new xDSL ops at the right place."""

    ret_idx: int = field(init=False)
    codegen_template: bool = field(init=False)
    curr_func: str = field(init=False)

    function_infos: Dict[str, FunctionInfo]

    def __init__(self, hint_converter: Dict[str, Any], function_infos: Dict[str, FunctionInfo]):
        inserter = OpInserter()
        
        self.globals = hint_converter.globals
        self.type_manager = TypeManager(inserter)
        self.induction_vars = dict()
        self.hint_converter = hint_converter
        self.symbol_table = dict()
        self.symbol_idx = dict()
        self.inserter = inserter
        self.function_infos = function_infos
        self.ret_idx = None
        self.curr_func = None
        self.codegen_template = False

    def visit(self, node: ast.AST):
        return super().visit(node)

    def generic_visit(self, node: ast.AST):
        raise CodegenInternalException(f"visitor for node {node} does not exist")

    def visit_Expr(self, node: ast.Expr):
        self.visit(node.value)

    def _cast(self, dst_ty: Attribute, value_ty: Attribute, value: SSAValue | Operation):
        # There are a lot of different casts, for now just put a placeholder op instead
        # of arith.trunci and friends.
        # TODO: implement casts.
        self.inserter.insert_op(unimplemented.Cast.get(value, dst_ty))
        return self.inserter.get_operand()
    
    def visit_Assert(self, node: ast.Assert):
        self.visit(node.test)
        msg = node.msg if node.msg is not None else ""
        op = cf.Assert.get(self.inserter.get_operand(), msg)
        self.inserter.insert_op(op)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """
        Visits type-annotated assignment operation, e.g.

        a: i32 = 3
        """
        # Make sure the symbol table knows the type information. For now we only allow
        # referring to the symbols within the function, but in future that should change!
        # TODO: fix symbol table.
        # self.symbol_table[node.target.id] = lhs_ty

        line, ty = self.symbol_table[node.target.id][0]
        if node.lineno == line:
            lhs_ty = ty
        elif node.lineno > line:
            self.symbol_table[node.target.id].pop(0)
            line, ty = self.symbol_table[node.target.id][0]
            assert line == node.lineno
            self.symbol_idx[node.target.id] += 1
            lhs_ty = ty
        else:
            raise Exception("something went wrong with inference!")

        # First, find the type of the LHS based on the type hint and create a new
        # symref declaration.
        # lhs_ty = self.hint_converter.convert_hint(node.annotation)

        symbol_name = "{}{}".format(node.target.id, self.symbol_idx[node.target.id])
        declare_op = symref.Declare.get(symbol_name)
        self.inserter.insert_op(declare_op)

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

        update_op = symref.Update.get(symbol_name, rhs)
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
            # Check if this is an assignemnt.
            #line, ty = self.symbol_table[node.targets[0].id][0]
            if node.lineno == self.symbol_table[node.targets[0].id][0][0]:
                ty = self.symbol_table[node.targets[0].id][0][1]
                symbol_name = "{}{}".format(node.targets[0].id, self.symbol_idx[node.targets[0].id])
                declare_op = symref.Declare.get(symbol_name)
                self.inserter.insert_op(declare_op)
                lhs_ty = ty

            elif len(self.symbol_table[node.targets[0].id]) > 1 and node.lineno == self.symbol_table[node.targets[0].id][1][0]:
                # This means this is an assignemt!
                self.symbol_table[node.targets[0].id].pop(0)
                ty = self.symbol_table[node.targets[0].id][0][1]
                self.symbol_idx[node.targets[0].id] += 1

                symbol_name = "{}{}".format(node.targets[0].id, self.symbol_idx[node.targets[0].id])
                declare_op = symref.Declare.get(symbol_name)
                self.inserter.insert_op(declare_op)
                lhs_ty = ty

            else:
                lhs_ty = self.symbol_table[node.targets[0].id][0][1]
            # self.state.inferred_type = lhs_ty

            # TODO: check types and add implicit casts if necessary.
            rhs = self.type_manager.match(lhs_ty, rhs)

            symbol_name = "{}{}".format(node.targets[0].id, self.symbol_idx[node.targets[0].id])
            update_op = symref.Update.get(symbol_name, rhs)
            self.inserter.insert_op(update_op)
            return
        
        if isinstance(node.targets[0], ast.Subscript):
            node = node.targets[0]
            indices: List[SSAValue] = []
            while isinstance(node, ast.Subscript):
                self.visit(node.slice)
                index = self.inserter.get_operand()
                index = self.type_manager.match(builtin.IndexType(), index)
                indices.append(index)
                node = node.value
        
            indices = list(reversed(indices))
            self.visit(node)
            indexed_value = self.inserter.get_operand()

            frontend_type = self.hint_converter.frontend_type_cache[indexed_value.typ.__class__]

            # TODO: check types and add implicit casts if necessary.
            if isinstance(indexed_value.typ, builtin.TensorType):
                rhs = self.type_manager.match(indexed_value.typ.element_type, rhs) 

            resolver = OpResolver.resolve_op_overload("__setitem__", frontend_type)
            if resolver is None:
                raise CodegenInternalException("operator __setitem__() is not supported")

            op = resolver()(rhs, indexed_value, *indices)
            self.inserter.insert_op(op)

            new_tensor = self.inserter.get_operand()
            update_op = symref.Update.get("{}{}".format(node.id, self.symbol_idx[node.id]), new_tensor)
            self.inserter.insert_op(update_op)

    def visit_comprehension(self, node: ast.comprehension):
        if len(node.ifs) > 0:
            raise CodegenInternalException("comprehension with if statement is not supported")

    def visit_ListComp(self, node: ast.ListComp):
        # Assume that target of generation is always a tensor.
        tensor_dims = []

        # Gather all necessary information from comprehension.
        local_ind_vars = []
        bounds = []
        for generator in node.generators:
            if not isinstance(generator, ast.comprehension):
                raise CodegenInternalException("every generator must be ast.comprehension class")

            if len(generator.ifs) > 0:
                raise CodegenInternalException("ni ifs in generator are supported")
            
            # Every target id is induction variable used in generation.
            if generator.target.id != "_":
                local_ind_vars.append(generator.target.id)

            if not isinstance(generator.iter, ast.Call) or not isinstance(generator.iter.func, ast.Name) or generator.iter.func.id != "range" or len(generator.iter.args) != 1:
                raise CodegenInternalException("this list comprehension is not supported")
        
            if isinstance(generator.iter.args[0], ast.Constant):
                dim = int(generator.iter.args[0].value)
            else:
                # Dynamically shaped tensor!
                dim = -1
                self.visit(generator.iter.args[0])
                op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
                bounds.append(op)
            tensor_dims.append(dim)
        
        has_static_shape = len(list(filter(lambda d: d == -1, tensor_dims))) == 0

        # Static shape is just a tensor splat.
        if has_static_shape:
            try:
                self.visit(node.elt)
                constant = self.inserter.get_operand()
                splat_op = tensor.Splat.get(constant, tensor_dims)
                self.inserter.insert_op(splat_op)
            except:
                # TODO: in this case we have something like this:
                # = [i + 2 for i in range(3)]
                # Ideally, we eant to convert it to tensor.from_elements
                # as the list comprehension is just [2, 3, 4]. One idea would
                # be to execute list comptehension to get the list and simply
                # process that AST :)
                raise CodegenInternalException("Unsupported list comprehension!")
        else:
            # Otherwise it is a tensor generate.
            prev_insertion_point = self.inserter.ip

            # Record induction variables.
            block = Block()
            for i, var in enumerate(local_ind_vars):
                block.insert_arg(builtin.IndexType(), i)
                self.induction_vars[var] = block.args[i]
            self.inserter.set_insertion_point_from_block(block)

            # Make sure we register the index type conversion.
            import xdsl.frontend.dialects.builtin as frontend_builtin
            if builtin.IndexType().__class__ not in self.hint_converter.frontend_type_cache:
                self.hint_converter.frontend_type_cache[builtin.IndexType().__class__] = frontend_builtin.IndexType().__class__

            # Visit the actual generated value.
            self.visit(node.elt)
            result = self.inserter.get_operand()
            el_ty = result.typ
            self.inserter.insert_op(tensor.Yield.get(result))

            # Create tensor.generate operation and insert it.
            body_region = Region.from_block_list([block])
            self.inserter.set_insertion_point_from_block(prev_insertion_point)

            op = tensor.Generate.from_region(bounds, body_region, tensor_dims, el_ty)
            self.inserter.insert_op(op)

            for var in local_ind_vars:
                self.induction_vars.pop(var)


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

        # Try to resolve this binary operator.
        op_name: str = node.op.__class__.__name__
        frontend_type = self.hint_converter.frontend_type_cache[lhs.typ.__class__]
        resolver = OpResolver.resolve_op_overload(op_name, frontend_type)
        if resolver is None:
            raise CodegenInternalException(f"binary operator {op_name} is not supported")
        
        # If resolved, we should get a binary op.
        op = resolver()(lhs, rhs)
        self.inserter.insert_op(op)

    def visit_Compare(self, node: ast.Compare):
        """
        Visits a comparison operation.
        """
        # First, allow a single comparison only.
        if len(node.comparators) != 1 or len(node.ops) != 1:
            raise CodegenInternalException(f"require a single comparator and op, found {len(node.comparators)} and {len(node.ops)}")

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
        frontend_type = self.hint_converter.frontend_type_cache[lhs.typ.__class__]
        resolver = OpResolver.resolve_op_overload(op_name, frontend_type)
        if resolver is None:
            raise CodegenInternalException(f"comparison operator {op_name} is not supported")

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
            raise CodegenInternalException(f"comparison operator {op_name} is not supported")
        
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
        names: List[(int, str)] = []
        for arg in reversed(node.args):
            # TODO: whet if I pass a list comprehension? How do side-effects work?
            if isinstance(arg, ast.Name):
                names.append(arg.id)
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
        # TODO: drop function table since we have self.functions now.
        if isinstance(node.func, ast.Name) and node.func.id in self.function_infos:
            callee_operand_types = [arg_info.xdsl_type for arg_info in self.function_infos[node.func.id].arg_info]

            # Check operand types.
            for i in range(num_args):
                actual_ty = operand_types[i]
                expected_ty = callee_operand_types[i]
                if actual_ty != expected_ty:
                    # TODO: implicit cast
                    operands[i] = self.type_manager.match(expected_ty, operands[i])
                    # raise CodegenInternalException(f"wrong argument type at position {i} when calling '{node.func.id}', expected {prettify(expected_ty)}, got {prettify(actual_ty)}")

            # Operand types match, so we can create a call operation and insert
            # it in the current block.
            outs = [return_info.xdsl_type for return_info in self.function_infos[node.func.id].return_info]
            call_op = func.Call.get(node.func.id, operands, outs) 
            self.inserter.insert_op(call_op)

            # Make sure we update side-effect arguments.
            # TODO: for nowwe assume these are last:
            # num_side_effects = len(self.side_effects[node.func.id])
            # for i in range(num_side_effects - 1, -1, -1):
            #     # TODO: assume all side-effect args are the last ones!
            #     name = names.pop()
            #     op = list(reversed(call_op.results))[i]
            #     symbol_name = "{}{}".format(name, self.symbol_idx[name])
            #     update_op = symref.Update.get(symbol_name, op)
            #     self.inserter.insert_op(update_op)
            return

        # Otherwise, get the module and the function names.
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            module_name = self.globals[func_name].__module__
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = node.func.attr
            module_name = self.globals[node.func.value.id].__name__
        else:
            raise CodegenInternalException("function calls are supported only in the form M.F() or F()")

        resolver = OpResolver.resolve_method(module_name, func_name)
        if resolver is None:
            # TODO: what about other standard libraries, max, sum, etc? They should be
            # handled here.
            raise CodegenInternalException(f"unknown function {func_name} from {module_name}")
        
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
            raise CodegenInternalException(f"unknown constant {node.value} of type {type(node.value)}")

    def check_for_loop_valid(self, node: ast.For):
        """Aborts if this loop cannot be lowered to xDSL or MLIR."""

        # Make sure we do not support Python hackery like:
        # for x in xs:      for x1, x2, x3 in xs:
        #   ...         or    ...
        # else:
        #   ...
        if len(node.orelse) > 0:
            raise CodegenInternalException(f"unexpected else clause in for loop on line {node.lineno}")
        if not isinstance(node.target, ast.Name):
            raise CodegenInternalException(f"expected a single induction target variable, found multiple in for loop on line {node.lineno}")

        # In xDSL/MLIR we can only have range-based loops as there is no concept of iterator.
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range" and 1 <= len(node.iter.args) <= 3:
            return
        raise CodegenInternalException(f"not a range-based loop on line {node.lineno}")


    def is_affine_for_loop(self, node: ast.For) -> bool:
        """Returns true if this for loop is affine."""
        for arg in node.iter.args:
            if not isinstance(arg, ast.Constant):
                return False
        return True


    def codegen_affine_for_loop(self, node: ast.For):
        """Gnereates xDSL for affine for loops."""

        # First, proces range arguments which should simply constants.
        args = node.iter.args
        start = 0
        end = 0
        step = 1
        if len(args) == 1:
            end = int(args[0].value)
        elif len(args) == 2:
            start = int(args[0].value)
            end = int(args[1].value)
        else:
            start = int(args[0].value)
            end = int(args[1].value)
            step = int(args[2].value)

        # Save previous insertion point.
        prev_insertion_point = self.inserter.ip

        entry_block = Block()
        entry_block.insert_arg(builtin.IndexType(), 0)
        self.induction_vars[node.target.id] = entry_block.args[0]

        body_region = Region.from_block_list([entry_block])

        # Create affine.for operation and insert it.
        op = affine.For.from_region([], start, end, body_region, step)
        self.inserter.set_insertion_point_from_block(prev_insertion_point)
        self.inserter.insert_op(op)
        self.inserter.set_insertion_point_from_block(entry_block)

        # Generate xDSL for the loop body.
        for stmt in node.body:
            self.visit(stmt)
        self.inserter.insert_op(affine.Yield.get())

        # Reset insertion point back. 
        self.inserter.set_insertion_point_from_block(prev_insertion_point)
        self.induction_vars.pop(node.target.id)


    def codegen_scf_for_loop(self, node: ast.For):
        args = node.iter.args
        if len(args) == 1:
            start_op = arith.Constant.from_int_and_width(0, builtin.IndexType())
            self.inserter.insert_op(start_op)
            self.visit(args[0])
            end_op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
            step_op = arith.Constant.from_int_and_width(1, builtin.IndexType())
            self.inserter.insert_op(step_op)
        elif len(args) == 2:
            self.visit(args[0])
            start_op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
            self.visit(args[1])
            end_op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
            step_op = arith.Constant.from_int_and_width(1, builtin.IndexType())
            self.inserter.insert_op(step_op)
        else:
            self.visit(args[0])
            start_op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
            self.visit(args[1])
            end_op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
            self.visit(args[2])
            step_op = self.type_manager.match(builtin.IndexType(), self.inserter.get_operand())
        
        # Save previous insertion point.
        prev_insertion_point = self.inserter.ip

        entry_block = Block()
        entry_block.insert_arg(builtin.IndexType(), 0)
        self.induction_vars[node.target.id] = entry_block.args[0]
        body_region = Region.from_block_list([entry_block])

        # Create scf.for operation and insert it.
        op = scf.For.from_region(start_op, end_op, step_op, [], body_region)
        self.inserter.set_insertion_point_from_block(prev_insertion_point)
        self.inserter.insert_op(op)
        self.inserter.set_insertion_point_from_block(entry_block)

        # Generate xDSL for the loop body.
        for stmt in node.body:
            self.visit(stmt)
        self.inserter.insert_op(scf.Yield.get())

        # Reset insertion point back. 
        self.inserter.set_insertion_point_from_block(prev_insertion_point)
        self.induction_vars.pop(node.target.id)

    def visit_For(self, node: ast.For):
        """Visits a for loop and creates scf.for or affine.for operation."""

        # First, check if this loop can be lowered to xDSL.
        self.check_for_loop_valid(node)

        assert isinstance(node.target, ast.Name)

        import xdsl.frontend.dialects.builtin as frontend_builtin
        if builtin.IndexType().__class__ not in self.hint_converter.frontend_type_cache:
            self.hint_converter.frontend_type_cache[builtin.IndexType().__class__] = frontend_builtin.IndexType().__class__

        # Next, we have to check if the loop is affine: for now we simply
        # check if all range arguments are constants. If not, we have to generate scf.for
        if self.is_affine_for_loop(node):
            self.codegen_affine_for_loop(node)
        else:
            self.codegen_scf_for_loop(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visits a function definition, e.g.

        def foo():
            ...
        """
        # TODO: this can be nicer!
        self.curr_func = node.name

        # Type inference!
        self.symbol_table = TypeInference(self.globals, self.function_infos).run_on_function(node)
        self.symbol_idx = dict()
        for k, v in self.symbol_table.items():
            self.symbol_idx[k] = 0

        # Then, convert type in the function signature.
        arg_types = [arg_info.xdsl_type for arg_info in self.function_infos[node.name].arg_info]
        return_types = [return_info.xdsl_type for return_info in self.function_infos[node.name].return_info]

        # TODO: This can be gated by if.
        side_effects = [arg_info.xdsl_type for arg_info in self.function_infos[node.name].arg_info if arg_info.has_side_effects]
        #return_types += side_effects

        # Create a region for the function body and entry block.
        entry_block = Block()
        body_region = Region.from_block_list([entry_block])
        func_op = func.FuncOp.from_region(node.name, arg_types, return_types, body_region)
        self.inserter.insert_op(func_op)
        self.inserter.set_insertion_point_from_block(entry_block)

        # Reset the symbol table.
        # TODO: this doesn't handle global variables at the moment.
        # self.symbol_table = dict()

        # All arguments are declared using symref.
        for i, arg in enumerate(node.args.args):
            symbol_name = "{}{}".format(arg.arg, self.symbol_idx[arg.arg])
            arg = entry_block.insert_arg(arg_types[i], i)
            entry_block.add_op(symref.Declare.get(symbol_name))
            # self.symbol_table[symbol_name] = arg_types[i]
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
                raise CodegenInternalException(f"expected 1 return type, got 0 in function {func_name}")
            self.inserter.insert_op(func.Return.get())

        # Move on with code generation for the next operation and record
        # the function to allow calls to it.
        self.inserter.set_insertion_point_from_op(func_op.parent_op())
        self.curr_func = None

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
        if node.id in self.induction_vars:
            self.inserter.stack.append(self.induction_vars[node.id])
            return 

        # TODO: make this look nicer, but name always uses the first!
        
        ty = self.symbol_table[node.id][0][1]
        symbol_name = "{}{}".format(node.id, self.symbol_idx[node.id])
    
        # if self.ret_idx is not None:
        #     # TODO: this is ugly, rework.
        #     for i, arg_info in enumerate(self.function_infos[self.curr_func].arg_info):

        #     for j, (i, xdsl_ty) in self.side_effects[self.curr_func]:
        #         if i == self.ret_idx:
        #             ty = xdsl_ty
        #             symbol_name = "{}{}".format(node.id, 0)
        #             break
        fetch_op = symref.Fetch.get(symbol_name, ty)
        self.inserter.insert_op(fetch_op)

    def visit_Tuple(self, node: ast.Tuple):
        # TODO: For now it s only needed in the return statement.
        assert self.ret_idx is not None

        for elt in node.elts:
            self.visit(elt)
            self.ret_idx += 1

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
            raise CodegenInternalException("return statement should be placed only at the end of the function body")

        func_name = parent_op.attributes["sym_name"].data
        func_return_types = parent_op.function_type.outputs.data

        if node.value is None:
            # Return nothing, check function signature matches.
            if len(func_return_types) != 0:
                raise CodegenInternalException(f"expected non-zero return types, got 0 in function '{func_name}'")
            return_op = func.Return.get()
            self.inserter.insert_op(return_op)
        else:
            # Return some type, check function signature matches as well.
            self.ret_idx = 0
            self.visit(node.value)
            self.ret_idx = None
            if isinstance(node.value, ast.Tuple):
                operands = []
                for _ in range(len(node.value.elts)):
                    op = self.inserter.get_operand()
                    operands.append(op)
                operands = list(reversed(operands))
            else:
                operands = [self.inserter.get_operand()]

            if len(func_return_types) == 0:
                raise CodegenInternalException(f"expected 0 return types, got more in function {func_name}")

            for i in range(len(operands)):
                if func_return_types[i] != operands[i].typ:
                    
                    if isinstance(operands[i].typ, builtin.TensorType) and operands[i].typ.element_type == func_return_types[i].element_type and len(list(filter(lambda d: d == -1, operands[i].typ.shape.data))) == 0:
                        # TODO: make this nicer. Basically, we better change the return type instead of adding cast if this
                        # is a dynamically sized tensor.
                        # HACK!
                        ops_before = [op for j, op in enumerate(parent_op.function_type.outputs.data) if j < i]
                        ops_after = [op for j, op in enumerate(parent_op.function_type.outputs.data) if j > i]
                        new_ops = ops_before + [operands[i].typ] + ops_after
                        object.__setattr__(parent_op.function_type.outputs, "data", new_ops)
                        # TODO: this has an implication on self.functions. It is better to put this into a separate pass.
                        # self.functions[enclosing_func][1][0] = operand.typ
                    else:
                        # TODO: implicit cast
                        operands[i] = self.type_manager.match(func_return_types[i], operands[i])

            return_op = func.Return.get(*operands)
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
                raise CodegenInternalException(f"expected non-zero return types, got 0 in function '{func_name}'")

            self.inserter.insert_op(func.Return.get())

    def visit_Subscript(self, node: ast.Subscript):
        """
        Visits subscript expressions like x[i][j][k]
        """
        indices: List[SSAValue] = []
        while isinstance(node, ast.Subscript):
            self.visit(node.slice)
            index = self.inserter.get_operand()
            index = self.type_manager.match(builtin.IndexType(), index)
            indices.append(index)
            node = node.value
        
        indices = list(reversed(indices))
        self.visit(node)
        indexed_value = self.inserter.get_operand()

        frontend_type = self.hint_converter.frontend_type_cache[indexed_value.typ.__class__]

        resolver = OpResolver.resolve_op_overload("__getitem__", frontend_type)
        if resolver is None:
            raise CodegenInternalException("operator __getitem__() is not supported")

        op = resolver()(indexed_value, *indices)
        self.inserter.insert_op(op)
