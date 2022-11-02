import ast
import inspect
import logging
from typing import Iterable, Type, Optional, List, Any, Dict

from dataclasses import dataclass
# from heco.dialects.fhe import SecretType
# from heco.frontend.dialects.fhe import Constant
from xdsl.dialects import builtin, cf, func, symref, arith, affine
from xdsl.dialects.irdl import Attribute
from xdsl.frontend_deprecated.block import block

from xdsl.frontend_deprecated.visitors.state import ProgramState
from xdsl.frontend_deprecated.visitors.utils import (
    get_xdsl_obj, has_type_of_python_type, is_frontend_obj, is_module,
    is_region, node_is_frontend_obj, py_hint_to_xdsl_type, VisitorException,
    get_xdsl_op
)
from xdsl.ir import Data, Operation, Block, SSAValue


@dataclass
class UnknownNode(Exception):
    t: Type

    def __str__(self) -> str:
        return f"NYI: unknown node type {self.t}"


_LineType = Operation | Block | SSAValue | List[Operation]


class FrontendToXDSL(ast.NodeVisitor):
    """Translates frontend program to MLIR.

    This visitor parses the Python AST of the frontend program, replaces
    all frontend Python objects with their corresponding xDSL
    attributes/operations.

    :raises UnknownNode: when it encounters Python AST nodes that for which
        no translation is implemented (yet).
    :raises VisitorException: when the translation of nodes fails.
    :return: xDSL module of the program.
    """

    result_type_stack: List[Type] = []

    def __init__(self, glob: Dict[str, Any], state: ProgramState,
                 logger: Optional[logging.RootLogger] = None) -> None:
        self.glob = glob
        self.state = state

        if not logger:
            logger = logging.getLogger("frontend_to_xdsl_visitor_logger")
            logger.setLevel(logging.INFO)
        self.logger = logger

    #
    # Utils
    #

    def _assert_target_is_supported(self, targets: Iterable[ast.AST]) -> bool:
        for target in targets:
            if not isinstance(target, ast.Name):
                raise VisitorException(
                    f"unsupported target '{ast.unparse(target)}' on line {target.lineno}.")

    def _return(self, op: Operation) -> Operation:
        self.state.add_ops(op)
        return op

    def _attr_to_constant(self, attr: Attribute, typ: Attribute = None, lineno: Optional[int] = None):
        if not typ:
            if not hasattr(attr, "typ"):
                err = f"Cannot derive type of index value {attr} for subscript expression."
                if lineno:
                    err += f" on line {lineno}"
                raise VisitorException(err)
            typ = attr.typ

        return self._return(arith.Constant.from_attr(attr, typ))

    #
    # General
    #

    def visit(self, node: Optional[ast.AST]) -> _LineType:
        if node is None:
            raise UnknownNode("Received None when expecting a valid node.")
        else:
            return super().visit(node)

    def generic_visit(self, node: ast.AST) -> _LineType:
        raise UnknownNode(
            f"translation of type {type(node)} not yet implemented.")

    #
    # Statements
    #

    def visit_AnnAssign(self, stmt: ast.AnnAssign) -> _LineType:
        self._assert_target_is_supported([stmt.target])

        type_hint = py_hint_to_xdsl_type(stmt.annotation, self.glob)

        result_type_stack_old = self.result_type_stack[::]
        self.result_type_stack.append(type_hint)
        if hasattr(type_hint, "parameters"):
            for type_hint_arg in type_hint.parameters:
                self.result_type_stack.append(type_hint_arg)
        value = self.visit(stmt.value)
        if isinstance(value, Attribute):
            value = self._attr_to_constant(value, type_hint)
        self.result_type_stack = result_type_stack_old

        # TODO: Currently we throw an error on assignments with duplicate type hints.
        self.state.add_variable(stmt.target.id, type_hint)

        declare_op = symref.Declare.get(stmt.target.id)
        update_op = symref.Update.get(stmt.target.id, value)

        # The value is already added to the current block while parsing it
        return self._return([declare_op, update_op])

    def visit_Assign(self, node: ast.Assign) -> _LineType:
        targets = node.targets
        if not hasattr(targets, "__iter__"):
            targets = [targets]

        self._assert_target_is_supported(targets)

        if len(targets) > 1:
            raise VisitorException(
                "NYI: multiple targets in assignment are not yet supported.")

        # The values are already added to the current block while parsing them
        value = self.visit(node.value)

        ops = []
        # TODO: multiple targets are not yet supported, see TODO further below.
        for target in targets:
            var_type, block_arg = self.state.lookup_variable(target.id)

            if block_arg:
                raise VisitorException(
                    f"Overwriting the block argument {target.id} is not allowed in the frontend.")

            if not var_type:
                # TODO: derive type from RHS if possible. Currently, this forces the first assignment to
                #   a variable v to be: `v: type = value`
                raise VisitorException(f"Encountered unknown variable name '{target.id}'. "
                                       "In case this is the first use of the variable, "
                                       "please annotated it's type.")

            # TODO: missing for multi-target support: split value into multiple values too
            #   and assign them to respective targets.
            if isinstance(value, Attribute):
                value = self._attr_to_constant(value, var_type)

            ops.append(symref.Update.get(target.id, value))

        return self._return(ops)

    #
    # Expressions
    #
    def visit_BinOp(self, node: ast.BinOp) -> _LineType:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(left, Attribute):
            left = self._attr_to_constant(left)
        if isinstance(right, Attribute):
            right = self._attr_to_constant(right)

        match node.op.__class__.__name__:
            case "Add":
                opname = "__add__"
            case "Sub":
                opname = "__sub__"
            case "Mult":
                opname = "__mul__"
            case "Div":
                opname = "__div__"
            case _:  # TODO NYI: MatMult, Mod, LShift, RShift, BitOr, BitXor, BitAnd, FloorDiv, Pow
                raise VisitorException(
                    f"NYI: operation {node.op.__class__.__name__} is not yet supported.")

        # Resolve the operation based on the left operand
        if isinstance(left, Operation):
            if len(left.results) != 1:
                raise VisitorException(
                    f"Unexpected number of results for operand of BinOp {opname}, expected 1, got {len(left.results)}.")
            typ = left.results[0].typ
        else:
            typ = left

        # TODO: Currently, this assumes all operators are called on the left operand.
        #   We should look into Python operator precedence to check whether this is actually the case.
        #   cf. NodeVisitor parsing for BinOp defines a precedence map with different types of precedence
        #   for different operators.
        xdsl_op = get_xdsl_op(typ.__class__.__name__, opname, self.glob)

        # TODO: this is a somewhat hacky solution to detect whether an operation has a
        #   default result type or not. Non-binary operations could have more arguments,
        #   but they don't overwrite operators.
        if "result_type" in inspect.signature(xdsl_op.get).parameters:
            bin_op = xdsl_op.get(left, right, result_type=typ)
        else:
            bin_op = xdsl_op.get(left, right)
        return self._return(bin_op)

    def visit_Call(self, call_op: ast.Call) -> _LineType:
        if isinstance(call_op.func, ast.Name) \
                and self.state.has_block_with_label(call_op.func.id):
            # Translate this function call to a block branch
            block = self.state.enter_block(call_op.func.id)
            return self._return(cf.Branch.get(block))

        fn = node_is_frontend_obj(call_op.func, self.glob)

        if fn:
            # Replace frontend function with xDSL function
            xdsl_fn = get_xdsl_obj(fn, self.glob)

            # Parse arguments like in super(), then append the type from the type hint as argument
            args = [self.visit(arg) for arg in call_op.args]

            # Try to derive type
            if len(self.result_type_stack) > 0:
                # Use type hint of LHS
                typ = self.result_type_stack.pop()
            elif hasattr(fn, "_default_typ"):
                # Use default value if available and no type hint is given
                typ = fn._default_typ
            else:
                # Try to derive the type from the first argument
                try:
                    typ = fn._default_type_from_args(args)
                except Exception as e:
                    raise VisitorException(f"cannot derive type of frontend object {ast.unparse(call_op)} "
                                           f"on line {call_op.lineno}. Please add type hints.\n(Exception: {e})")

            if issubclass(xdsl_fn, Data):
                if len(call_op.args) != 1:
                    raise VisitorException(
                        "Unexpected number of arguments, the frontend expects data objects to have a single argument.")

                arg = call_op.args[0]
                if not isinstance(arg, ast.Constant):
                    raise VisitorException(
                        "Data objects only takes constant arguments for now.")

                ret_obj = xdsl_fn(arg.value)
            elif issubclass(xdsl_fn, Attribute):
                # TODO: assumes that the type is always the last parameter in the xDSL dialect
                #   We should enforce this in xDSL.
                ret_obj = xdsl_fn([*args, typ])
            elif issubclass(xdsl_fn, Operation):
                ret_obj = xdsl_fn.build(operands=args, result_types=[typ])
            else:
                raise VisitorException(
                    f"Failed to translate unsupported frontend function {fn}.")

            return ret_obj
        else:
            raise VisitorException(
                "NYI: calling functions defined in the frontend program is not yet supported.")

    def visit_Constant(self, node: ast.Constant) -> _LineType:
        # TODO: use default translation specified by frontend developer or derive types if possible.
        if isinstance(node.value, int):
            return builtin.IntegerAttr.from_int_and_width(node.value, 64)
        if isinstance(node.value, float):
            return builtin.FloatData(node.value)

        raise VisitorException(
            f"NYI: no translation for {type(node.value)} constant implemented.")

    def visit_For(self, node: ast.For) -> _LineType:
        if len(node.orelse) > 0:
            raise VisitorException(
                "NYI: the frontend does not yet support the orelse clause for for-loops.")

        if not isinstance(node.target, ast.Name):
            raise VisitorException(
                "NYI: for-loops currently only support a single target.")

        if not isinstance(node.iter, ast.Call) \
                or not (isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range") \
                or not 2 <= len(node.iter.args) <= 3 \
                or not all([isinstance(arg, ast.Constant) for arg in node.iter.args]):
            raise VisitorException("NYI: for-loops currently only support iterating over "
                                   "range with constant integer lower *and* upper bound "
                                   "(and optionally a step).")

        lower_bound = node.iter.args[0].value
        upper_bound = node.iter.args[1].value
        step = 1
        if len(node.iter.args) > 2:
            step = node.iter.args[2].value

        for_body_region = self.state.enter_op_region()
        for_body_block = self.state.enter_block()

        for_body_block.insert_arg(builtin.IndexType(), 0)
        self.state.add_variable(
            node.target.id, builtin.IndexType(), for_body_block.args[0])

        for stmt in node.body:
            self.visit(stmt)

        self.state.exit_block()
        self.state.exit_region()

        return self._return(affine.For.from_region([], lower_bound, upper_bound, for_body_region, step))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> _LineType:
        error_template = f"invalid arguments for function on line {node.lineno}." + \
            "Frontend functions/blocks currently only support normal arguments and type hints, {}" + \
            f"\nFunction:\n{ast.unparse(node)}"

        for a in ["vararg", "kwarg"]:
            if getattr(node.args, a) != None:
                raise VisitorException(
                    error_template.format("no vararg or kwarg."))

        for a in ["kwonlyargs", "kw_defaults", "defaults"]:
            if len(getattr(node.args, a)) > 0:
                raise VisitorException(error_template.format(
                    "no kwonlyargs, kw_defaults, or defaults."))

        arg_names = []
        arg_types = []
        for i, arg in enumerate(node.args.args):
            arg_names.append(arg.arg)

            ann = arg.annotation
            if not has_type_of_python_type(ann):
                raise VisitorException(
                    f"Missing type hint for argument {i} of function {node.name} on line {ann.lineno}.")
            arg_types.append(py_hint_to_xdsl_type(ann, self.glob))

        return_types = [py_hint_to_xdsl_type(node.returns, self.glob)]

        if self.state.has_block_with_label(node.name):
            block = self.state.enter_block(node.name)

            # Add block arguments to variables to replace them inside the block.
            for i, arg_type in enumerate(arg_types):
                block.insert_arg(arg_type, i)
                self.state.add_variable(arg_names[i], arg_type, block.args[i])

            for stmt in node.body:
                self.visit(stmt)

            self.state.exit_block()
            return block

        body_region = self.state.enter_op_region()
        block = self.state.enter_block()

        # Add function arguments to replace them in the function body with block arguments
        for i in range(len(arg_types)):
            block.insert_arg(arg_types[i], i)
            self.state.add_variable(arg_names[i], arg_types[i], block.args[i])

        # Parse function body
        for stmt in node.body:
            self.visit(stmt)

        self.state.exit_block()
        self.state.exit_region()

        return self._return(func.FuncOp.from_region(node.name, arg_types,
                                                    return_types, body_region))

    def visit_Name(self, name: ast.Name) -> _LineType:
        obj = is_frontend_obj(name)
        if obj:
            return get_xdsl_obj(obj)

        # Check whether the variable already exists in this scope
        var_type, block_arg = self.state.lookup_variable(name.id)

        # Replace variables for block arguments with reference to block argument
        if block_arg:
            return block_arg

        if var_type:
            return self._return(symref.Fetch.get(name.id, var_type))
        else:
            raise VisitorException(
                f"Encountered unknown variable name '{name.id}'")

    def visit_Return(self, node: ast.Return) -> _LineType:
        stmts = self.visit(node.value)
        return self._return(func.Return.get(stmts))

    def visit_With(self, node: ast.With) -> Operation:
        if is_region(node):
            # advance state to next region (must be before recursive visit, so that the blocks
            # are added to the correct region!)
            region = self.state.get_next_region()
            is_block = [isinstance(self.visit(stmt), Block)
                        for stmt in node.body]

            if not all(is_block) and any(is_block):
                raise VisitorException("Cannot mix Blocks and Operations in the same region. "
                                       "Either put all Operations into Blocks or only add Operations "
                                       "(which are added to an implicit Block).")

            return region
        elif is_module(node):
            # advance state to next module (must be before recursive visit, so that the region
            # is added to the correct module!)
            module = self.state.get_next_module()
            for stmt in node.body:
                self.visit(stmt)
            return module
        else:
            raise VisitorException(f"unsupported With-Statement encountered on line {node.lineno}: {ast.unparse(node)}"
                                   "The frontend currently only supports With statements for regions and modules.")

    def visit_Subscript(self, node: ast.Subscript) -> _LineType:
        idx = self.visit(node.slice)
        value = self.visit(node.value)

        if isinstance(value, Operation):
            if len(value.results) != 1:
                raise VisitorException("Subscript value must be an Operation/SSAValue with a "
                                       f"single result, got {len(value.results)} results.")
            name = value.results[0].typ.__class__.__name__
        elif isinstance(value, SSAValue):
            name = value.typ.__class__.__name__

        xdsl_op = get_xdsl_op(name, "__getitem__", self.glob)

        if isinstance(idx, Attribute):
            idx = self._attr_to_constant(idx, lineno=node.lineno)

        # TODO: The problem is obj is the class, not an instance of the class.
        #   However, this hack only works if obj has no arguments.
        return self._return(xdsl_op.get(value, idx))
