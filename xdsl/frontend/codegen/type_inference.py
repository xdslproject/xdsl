import ast

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type
from xdsl.dialects.builtin import FunctionType, i1, i64, f32, IndexType, IntegerType, TensorType
from xdsl.frontend.codegen.functions import FunctionInfo
from xdsl.frontend.codegen.type_conversion import TypeConverter
from xdsl.ir import Attribute


@dataclass
class TypeInference():
    """
    Type inference engine which:
      1) infers the right type of expressions.
      3)
    """

    globals: Dict[str, Any]

    function_infos: Dict[str, FunctionInfo]

    def run_on_function(self, node: ast.FunctionDef):
        visitor = TypeInferenceVisitor(self.globals, self.function_infos, node)
        for stmt in node.body:
            visitor.visit(stmt)
        # visitor.info()
        return visitor.types

class TypeInferenceVisitor(ast.NodeVisitor):

    function_infos: Dict[str, FunctionInfo] = field(init=False)

    type_converter: TypeConverter = field(init=False)

    types: Dict[str, List[Tuple[int, Type]]] = field(init=False)

    recurse: bool = False

    def __init__(self, globals: Dict[str, Any], function_infos: Dict[str, FunctionInfo], node: ast.FunctionDef):
        self.type_converter = TypeConverter(globals)
        self.function_infos = function_infos
        arg_info = function_infos[node.name].arg_info

        self.types = dict()
        for i, arg in enumerate(node.args.args):
            arg_name = arg.arg
            arg_type = arg_info[i].xdsl_type
            self.types[arg_name] = [(arg.lineno, arg_type)]
    
    def info(self):
        for v, ts in self.types.items():
            for i, info in enumerate(ts):
                line, t = info
                # print('line {:2}: {:6} |  {}'.format(line, "{}{}".format(v, i), prettify(t)))
    
    def kills_definition(self, src_ty: Attribute, dst_ty: Attribute) -> bool:
        if src_ty == dst_ty:
            return False

        if isinstance(src_ty, IndexType):
            if isinstance(dst_ty, IntegerType):
                return False

        if isinstance(src_ty, IntegerType):
            if isinstance(dst_ty, IndexType) or isinstance(dst_ty, IntegerType):
                return False

        if isinstance(src_ty, TensorType):
            if isinstance(dst_ty, TensorType):
                # tensor.cast
                if src_ty.element_type == dst_ty.element_type and src_ty.shape == dst_ty.shape:
                    return False

        return True

    def visit(self, node: ast.AST):
        return super().visit(node)

    def visit_For(self, node: ast.For):
        # TODO: not support this traversal.
        return
    
    def visit_If(self, node: ast.If):
        raise Exception("type inference not supported for conditionals!")

        # TODO: this is stupid but it works. Make sure we avoid copies.
        # types = deepcopy(self.types)
        # for stmt in node.body:
        #     self.visit(stmt)

        # true_types = self.types
        # self.types = deepcopy(types)
        # for stmt in node.orelse:
        #     self.visit(stmt)
        # false_types = self.types

        # for v, ts in types.items():
        #     for i, info in enumerate(ts):
        #         line, t = info
        #         print('line {:2}: {:6} |  {}'.format(line, "{}{}".format(v, i), prettify(t)))

        # print("---------------------------")

        # for v, ts in true_types.items():
        #     for i, info in enumerate(ts):
        #         line, t = info
        #         print('line {:2}: {:6} |  {}'.format(line, "{}{}".format(v, i), prettify(t)))

        # print("------------------------------------")

        # for v, ts in false_types.items():
        #     for i, info in enumerate(ts):
        #         line, t = info
        #         print('line {:2}: {:6} |  {}'.format(line, "{}{}".format(v, i), prettify(t)))

        # print("------------------------------------")

        # # Compare how the values changed.
        # true_types = self.types
        # for v in true_types.keys():
        #     if v in false_types and v not in types:
        #         # We have a new type!
        #         last_true_type = true_types[v][-1]
        #         last_false_type = false_types[v][-1]

        #         # TODO: here we should fail if types are mismatched.
        #         assert last_true_type == last_false_type

        #     if v in types:
        #         types[v] += true_types[v]
        #     else:
        #         types[v] = true_types[v]

        # for v in false_types.keys():
        #     if v in types:
        #         types[v] += false_types[v]
        #     else:
        #         types[v] = false_types[v]

        # self.types = types


    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Other AST nodes do not make sense.
        assert isinstance(node.target, ast.Name)
        
        var_name = node.target.id
        type = self.type_converter.convert_type_hint(node.annotation)

        if var_name not in self.types:
            self.types[var_name] = [(node.lineno, type)]
        else:
            # Otherwise, this new type is a new variable which kills previous
            # definition. 
            # TODO: is this a new variable? Or do we want different semantics?
            self.types[var_name].append((node.lineno, type))
    
    def visit_Name(self, node: ast.Name):
        if self.recurse:
            assert node.lineno >= self.types[node.id][-1][0]
            return self.types[node.id][-1][1] 

    def visit_Constant(self, node: ast.Constant):
        if self.recurse:
            # TODO: this is a copy from TypeManager, and has to be defined by the frontend
            # program ideally.
            default_types = {
                bool: i1,
                int: i64,
                float: f32,
                str: str, # TODO: what is this? Use for arith.cmp mnemonic, not sure if we want to support this.
            }

            expr_type = default_types[type(node.value)]
            return expr_type

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise Exception("More than one target!")

        # We only consider new variable assignemnts for now.
        # TODO: do we need subscripts though? Say we have a[i], then
        # having type check might not be that important, since the type
        # is inferred when a is created! Basically, this means that
        #
        #     a = [0 for i in range(10)]
        #
        # can be used without a type annotation, in general. On the contrary,
        #
        #     a = []
        #
        # cannot, because we do not know the type and have to infer based on the element
        # we put inside. But if we put both ints and say floats, it becomes undecidable
        # and we should error.
        if not isinstance(node.targets[0], ast.Name):
            return
        
        var_name = node.targets[0].id
        expr = node.value

        # Now let's infer the type of the expression. There are quite a bit of cases to consider.
        if isinstance(expr, ast.Constant) or isinstance(expr, ast.Name):
            self.recurse = True
            expr_type = self.visit(expr)
            self.recurse = False

        elif isinstance(expr, ast.ListComp):
            # TODO: we assume that list commprehension always generates a tensor.

            # Process the shape first.
            dims = []
            for generator in expr.generators:
                # TODO: Here we should do same cehcks as in codegen visitor. Probably we have to separate all checks
                # so that we terminate early and do not check on demand.
                if isinstance(generator.iter.args[0], ast.Constant):
                    dims.append(int(generator.iter.args[0].value))
                else:
                    dims.append(-1)                
            
            # Infer type of the element.
            self.recurse = True
            element_type = self.visit(expr.elt)
            self.recurse = False
            expr_type = TensorType.from_type_and_list(element_type, dims)

        elif isinstance(expr, ast.UnaryOp):
            # TODO: unary expressions do not change the type, in general. But if they do,
            # we should probaby teke some spec into account.
            self.recurse = True
            expr_type = self.visit(expr.operand)
            self.recurse = False

        elif isinstance(expr, ast.BinOp):
            # In MLIR/xDSL, binary operations usually take same type operands.
            # Assume this is the case, or at least that they can be casted to one another.
            self.recurse = True
            lhs_type = self.visit(expr.left)
            rhs_type = self.visit(expr.right)
            self.recurse = False
            
            # TODO: we should have a spec: op, lhs, rhs --> type, but for now we can just use
            # lhs, I think. In general, we want to support:
            # int op float is float, int op bool is int, etc. based on Python typing rules.
            assert lhs_type == rhs_type
            expr_type = lhs_type

        elif isinstance(expr, ast.Call):

            # TODO: can we have a call not to ast.Name?
            if expr.func.id in self.function_infos:
                # This is a known function.
                return_info = self.function_infos[expr.func.id].ret_info

                # Technically, here we can also type check arguments, e.g. passing ints as float
                # arguments, but again, this can be future work.
                # TODO: type check arguments.

                # We only support a single assignemnt at the moment.
                assert len(return_info) == 1
                expr_type = return_info[0].xdsl_type

            else:
                # Can be a standar library but let's not support this at the moment.
                # TODO: support standard library functions.
                raise Exception("cannot infer type from unknown function")

        elif isinstance(expr, ast.IfExp):
            self.recurse = True
            true_expr_type = self.visit(expr.body)
            false_expr_type = self.visit(expr.orelse)
            self.recurse = False

            # TODO: again, we need some kind of spec. What do we do when
            # we have '4 if condition else 0.33', is it int or float?
            assert true_expr_type == false_expr_type
            expr_type = lhs_type
            
        else:
            raise Exception("cannot infer type")

        if var_name not in self.types:
            self.types[var_name] = [(expr.lineno, expr_type)]
        else:
            # Check if this is re-assignemnt, or a new variable.
            assert expr.lineno >= self.types[var_name][-1][0]

            last_type = self.types[var_name][-1][1]
            if self.kills_definition(last_type, expr_type):
                self.types[var_name].append((node.lineno, expr_type))