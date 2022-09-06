from typing import List, Optional
from enum import Enum
from dataclasses import dataclass
from toy.Location import Location

INDENT = 2

@dataclass
class VarType:
    'A variable type with shape information.'
    shape: List[int]

class ExprASTKind(Enum):
    Expr_VarDecl = 1 
    Expr_Return = 2
    Expr_Num = 3
    Expr_Literal = 4
    Expr_Var = 5
    Expr_BinOp = 6
    Expr_Call = 7
    Expr_Print = 8

    
@dataclass(init=False)
class Dumper():
    lines: List[str]
    indentation: int
    
    def __init__(self, lines, indentation=0):
        self.lines = lines
        self.indentation = indentation
    
    def append(self, prefix, line):
        self.lines.append(' ' * self.indentation * INDENT + prefix + line)
    
    def append_list(self, prefix, exprs, block):
        self.append(prefix, '[')
        child = self.child()
        for expr in exprs:
            block(child, expr)
        self.append('', ']')
    
    def child(self):
        return Dumper(self.lines, self.indentation+1)
    
    @property
    def message(self):
        return '\n'.join(self.lines)

    
@dataclass
class ExprAST:
    loc: Location
    
    def __init__(self, loc):
        self.loc = loc
        print(self.dump())
    
    @property
    def kind(self):
        raise AssertionError(f'ExprAST kind not defined for {type(self)}')

    def _dump(self, prefix, dumper):
        dumper.append(prefix, self.__class__.__name__)
        
    def dump(self):
        dumper = Dumper([])
        self._dump('', dumper)
        return dumper.message
    
@dataclass
class VarDeclExprAST(ExprAST):
    'Expression class for defining a variable.'
    name: str
    varType: VarType
    expr: ExprAST
    
    @property
    def kind(self):
        return ExprASTKind.Expr_VarDecl
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        child.append('name: ', f'{self.name}')
        child.append('type: ', f'{self.varType}')
        self.expr._dump('expr: ', child)
    
@dataclass
class ReturnExprAST(ExprAST):
    'Expression class for a return operator.'
    expr: Optional[ExprAST]
    
    @property
    def kind(self):
        return ExprASTKind.Expr_Return
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        if self.expr is not None:
            child = dumper.child()
            self.expr._dump('expr: ', child)

@dataclass
class NumberExprAST(ExprAST):
    'Expression class for numeric literals like "1.0".'
    val: float
    
    @property
    def kind(self):
        return ExprASTKind.Expr_Num
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        child.append('val: ', f'{self.val}')

@dataclass
class LiteralExprAST(ExprAST):
    'Expression class for a literal value.'
    values: List[ExprAST]
    dims: List[int]
    
    @property
    def kind(self):
        return ExprASTKind.Expr_Literal
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        child.append_list('values: ', self.values, lambda dd, val: val._dump('', dd))
        child.append('dims: ', f'{self.dims}')
        
@dataclass
class VariableExprAST(ExprAST):
    'Expression class for referencing a variable, like "a".'
    name: str
    
    @property
    def kind(self):
        return ExprASTKind.Expr_Var
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        child.append('name: ', f'{self.name}')

@dataclass
class BinaryExprAST(ExprAST):
    'Expression class for a binary operator.'
    op: str
    lhs: ExprAST
    rhs: ExprAST
    
    @property
    def kind(self):
        return ExprASTKind.Expr_BinOp
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        child.append('op: ', self.op)
        self.lhs._dump('lhs: ', child)
        self.rhs._dump('rhs: ', child)

@dataclass
class CallExprAST(ExprAST):
    'Expression class for function calls.'
    callee: str
    args: List[ExprAST]
    
    @property
    def kind(self):
        return ExprASTKind.Expr_Call
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        child.append('callee: ', f'{self.callee}')
        child.append_list('args: ', self.args, lambda dd, arg: arg._dump('', dd))


@dataclass
class PrintExprAST(ExprAST):
    'Expression class for builtin print calls.'
    arg: ExprAST
    
    @property
    def kind(self):
        return ExprASTKind.Expr_Print
    
    def _dump(self, prefix, dumper):
        super()._dump(prefix, dumper)
        child = dumper.child()
        self.arg._dump('arg: ', child)
    
    
@dataclass
class PrototypeAST:
    '''
    This class represents the "prototype" for a function, which captures its
    name, and its argument names (thus implicitly the number of arguments the
    function takes).
    '''
    loc: Location
    name: str
    args: List[VariableExprAST]

    def dump(self):
        dumper = Dumper([])
        self._dump('', dumper)
        return dumper.message

    def _dump(self, prefix, dumper):
        dumper.append(prefix, self.__class__.__name__)
        child = dumper.child()
        child.append('name: ', f'{self.name}')
        child.append_list('args: ', self.args, lambda dd, arg: arg._dump('', dd))

@dataclass
class FunctionAST:
    'This class represents a function definition itself.'
    loc: Location
    proto: PrototypeAST
    body: List[ExprAST]

    def dump(self):
        dumper = Dumper([])
        self._dump('', dumper)
        return dumper.message

    def _dump(self, prefix, dumper):
        dumper.append(prefix, self.__class__.__name__)
        child = dumper.child()
        self.proto._dump('proto: ', child)
        child.append_list('body: ', self.body, lambda dd, stmt: stmt._dump('', dd))

@dataclass
class ModuleAST:
    'This class represents a list of functions to be processed together'
    funcs: List[FunctionAST]

    def dump(self):
        dumper = Dumper([])
        self._dump('', dumper)
        return dumper.message

    def _dump(self, prefix, dumper):
        dumper.append(prefix, self.__class__.__name__)
        child = dumper.child()
        child.append_list('functions: ', self.funcs, lambda dd, func: func._dump('', dd))