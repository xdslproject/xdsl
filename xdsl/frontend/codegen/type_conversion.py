import ast
import xdsl.dialects.builtin as builtin

from dataclasses import dataclass, field
from typing import _GenericAlias, Any, Dict, Optional, Type
from xdsl.frontend.codegen.exception import CodegenInternalException, CodegenException
from xdsl.frontend.dialects.builtin import Float32Type, FrontendType, TensorType, IntegerType, IndexType
from xdsl.ir import Attribute


@dataclass
class TypeConverter:
    """
    Class responsible for conversion of Python type hints to concrete xDSL
    types.
    """

    globals: Dict[str, Any]
    """
    Stores all globals in the current Python program, including imports. This is useful
    because we can lookup a class which corresponds to the type annotation.
    """

    type_cache: Dict[str, Attribute] = field(default_factory=dict)
    """
    Cache for xDSL types created so far to avoid repeated conversions.
    """

    frontend_type_cache: Dict[Attribute, type | FrontendType] = field(default_factory=dict)
    """
    Cache for front-end types which allows to get the source type based on xDSL type. This is
    useful if we want to see what operations does this xDSL type support, which is builtin on
    fronte-nd level.
    """

    def cache_type(self, frontend_type: type | FrontendType, frontend_type_name: str, xdsl_type: Attribute):
        """Records front-end and corresponding xDSL types in cache."""
        if frontend_type_name not in self.type_cache:
            self.type_cache[frontend_type_name] = xdsl_type
        if xdsl_type.__class__ not in self.frontend_type_cache:
            self.frontend_type_cache[xdsl_type.__class__] = frontend_type

    def __post_init__(self):
        # Cache python types.
        self.cache_type(IntegerType, "int", builtin.IntegerType.from_width(64))
        self.cache_type(IntegerType, "bool", builtin.IntegerType.from_width(1))
        self.cache_type(Float32Type, "float", builtin.Float32Type())
        
        # Cache index type because it is always used implicitly in loops, etc.
        self.cache_type(IndexType, "index", builtin.IndexType())

    def _convert_subscript(self, hint: ast.Subscript) -> Attribute:
        ty_name = hint.value.id
        ty = self.globals[ty_name]

        # TODO: this should also be defined by the frontend program.
        if ty_name == "List":
            # This is a dynamically sized tensor!
            num_dims = 1
            node = hint.slice
            while isinstance(node, ast.Subscript):
                num_dims += 1
                node = node.slice

            assert isinstance(node, ast.Name)
            el_ty = self._convert_name(node)
            xdsl_ty = builtin.TensorType.from_type_and_list(el_ty, [-1 for d in range(num_dims)])
            if xdsl_ty.__class__ not in self.frontend_type_cache:
                    self.frontend_type_cache[xdsl_ty.__class__] = TensorType
            return xdsl_ty


        # Any type hint must be a frontend type.
        if issubclass(ty, FrontendType):
            if isinstance(hint.slice, ast.Tuple):
                args = []
                for ty_arg in hint.slice.elts:
                    if isinstance(ty_arg, ast.Name):
                        xdsl_ty = self._convert_name(ty_arg)
                        args.append(xdsl_ty)
                    elif isinstance(ty_arg, ast.Subscript):
                        
                        # TODO: fix this porperly, but it should be shape!
                        ty_args = ty_arg.slice
                        res = []
                        if isinstance(ty_args, ast.Tuple):
                            for ty_arg in ty_args.elts:
                                if isinstance(ty_arg.slice, ast.UnaryOp):
                                    op_name = ty_arg.slice.op.__class__.__name__
                                    if op_name == "USub":
                                        # TODO: This is a dynamic shape! But we should implement this properly.
                                        assert int(ty_arg.slice.operand.value) == 1
                                        v = -int(ty_arg.slice.operand.value)
                                else:
                                    v = int(ty_arg.slice.value)
                                res.append(v)
                        args.append(res)
                    else:
                        msg = f"expected 'Name' or 'Subscript', got {ty_arg.__name__}"
                        raise CodegenInternalException(msg)
                
                xdsl_ty = ty.to_xdsl()(*args)
                if xdsl_ty.__class__ not in self.frontend_type_cache:
                    self.frontend_type_cache[xdsl_ty.__class__] = ty
                return xdsl_ty

            elif isinstance(hint.slice, ast.Name):
                xdsl_ty = ty.to_xdsl()(self._convert_name(hint.slice))
                if xdsl_ty.__class__ not in self.frontend_type_cache:
                    self.frontend_type_cache[xdsl_ty.__class__] = ty
                return xdsl_ty
            else:
                msg = f"expected 'Tuple', got {hint.slice}"
                raise CodegenInternalException(msg)

        # Otherwise abort.
        msg = f"expected a sublcass of FrontendType, got {hint.slice}"
        raise CodegenInternalException(msg)

    def _convert_name(self, type_hint: ast.Name) -> Attribute:
        # First, check if we have already converted this type hint. This also takes care
        # of primitive Python types like int, float, etc.
        type_name = type_hint.id
        if type_name in self.type_cache:
            return self.type_cache[type_name]

        # Otherwise, it must be some frontend type, and we can look up its class using the
        # imports.
        # TODO: we should restrict the lookup to front-end types only, i.e. those living under
        # 'xdsl.frontend' package. Hardcoding this seems not great, so we opt for subclassing
        # 'FrontendType'. But is this enough?
        type_class = self.globals[type_name]

        # First, type can be generic, e.g. IntegerType.
        if isinstance(type_class, _GenericAlias):
            args = []
            for type_arg in type_class.__args__:

                # TODO: This is enough to support simple cases like Literal[3]. At the moment, it is
                # only used by integers that have a generic bitwidth and signedness. So in theory here
                # we may want a visitor that convertes Literal[3] to 3, Tuple[Literal[1], Literal[2]]
                # to (1, 2), etc. For example, see deprecated front-end implementation for one.
                if len(type_arg.__args__) == 1 or isinstance(type_arg.__args__[0], int):
                    args.append(type_arg.__args__[0])
                    continue

                if len(type_arg.__args__) != 1:
                    raise CodegenException(type_hint.lineno, type_hint.col_offset, f"Expected 1 type argument for generic type {type_name}, got {len(type_arg.__args__)} types instead.")
                if not isinstance(type_arg.__args__[0], int):
                    raise CodegenException(type_hint.lineno, type_hint.col_offset, f"Expected 'int' type argument for generic type {type_name}, got '{type(type_arg.__args__[0])}' type instead.")

            # Finally, get the constructor of this type and build an xDSL type.
            if issubclass(type_class.__origin__, FrontendType):
                xdsl_type = type_class.to_xdsl()(*args)
                print(type_name)
                self.cache_type(type_class.__origin__, type_name, xdsl_type)
                return xdsl_type

            # If this is not a subclass of FrontendType, then abort.
            raise CodegenException(type_hint.lineno, type_hint.col_offset, f"Expected a sublcass of 'FrontendType', got {type_class.__origin__.__name__}.")

        # Otherwise, type can be a simple non-generic front-end type, e.g. 'IndexType(FrontendType)'.
        if issubclass(type_class, FrontendType):
            xdsl_type = type_class.to_xdsl()()
            self.cache_type(type_class, type_name, xdsl_type)
            return xdsl_type

        # Otherwise, abort.
        # TODO: any other corner cases?
        raise CodegenException(type_hint.lineno, type_hint.col_offset, f"Unknown type hint {type_class.__name__} for 'ast.Name' expression.")

    def convert_type_hint(self, type_hint: ast.expr) -> Optional[Attribute]:
        """Handles all type hint conversions."""

        # Type hint can be not provided, e.g. when returning None from the
        # function implicitly. Then simply return None and the caller should
        # decide what to do next.
        if type_hint is None:
            return None

        # In general, any type hint is a Subscript AST node, for example
        # Foo[Literal[2]].
        if isinstance(type_hint, ast.Subscript):
            return self._convert_subscript(type_hint)

        # Type hint can also be a TypeAlias. For example, one can define foo = Foo[Literal[2]].
        # This case also handle standard Python types, like int, float, etc.
        if isinstance(type_hint, ast.Name):
            return self._convert_name(type_hint)

        # In all other cases, abort.
        raise CodegenException(type_hint.lineno, type_hint.col_offset, f"Unknown type hint node {type_hint}.")
