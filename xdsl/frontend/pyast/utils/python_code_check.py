import ast

# Type aliases for simplicity.
BlockMap = dict[str, ast.FunctionDef]
FunctionData = tuple[ast.FunctionDef, BlockMap]
FunctionMap = dict[str, FunctionData]
