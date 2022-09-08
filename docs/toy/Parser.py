
from typing import List

from toy.Tokenizer import *
from toy.AST import *

class ParseError(Exception):
    def __init__(self, token, expected, context='', line=''):
        loc = token.loc
        
        message = f'Parse error ({loc.line}, {loc.col}): expected '
        if isinstance(expected, str):
            message += expected
        else:
            message += expected.__class__.__name__
        
        if len(context):
            message += ' ' + context
        
        message += f" but has Token '{token.text}'\n"
        
        if len(line):
            message += line + '\n'
        
        super().__init__(message)
        
    pass

class Parser:
    file: str
    program: str
    tokens: List[Token]
    pos: int
    
    def __init__(self, file, program):
        self.file = file
        self.program = program
        self.tokens = tokenize(file, program)
        self.pos = 0
    
    def getToken(self):
        return self.tokens[self.pos]
    
    def getTokenPrecedence(self):
        PRECEDENCE = {
            '-': 20,
            '+': 20,
            '*': 40,
        }
        op = self.getToken().text
        
        try:
            return PRECEDENCE[op]
        except:
            return -1

    def peek(self, pattern=None):
        token = self.getToken()
        tokenType, text = (None, pattern) if isinstance(pattern, str) else (pattern, None)
        if tokenType is not None:
            if type(token) is not tokenType:
                self.parseError(tokenType)
                
        if text is not None:
            if token.text != text:
                self.parseError(f"'{text}'")

    def check(self, pattern=None):
        try:
            self.peek(pattern)
            return True
        except ParseError:
            return False
        
    def pop(self, pattern=None):
        self.peek(pattern)
        self.pos += 1
        return self.tokens[self.pos-1]
    
    def parseModule(self):
        'Parse a full Module. A module is a list of function definitions.'
        functions = []
        
        while not self.check(EOFToken):
            functions.append(self.parseDefinition())
        
        # If we didn't reach EOF, there was an error during parsing
        self.pop(EOFToken)
        
        return ModuleAST(functions)
    
    def parseReturn(self):
        '''
        Parse a return statement.
        return :== return ; | return expr ;
        '''
        returnToken = self.pop('return')
        expr = None
        
        # Return takes an optional argument
        if not self.check(';'):
            expr = self.parseExpression()
        
        return ReturnExprAST(returnToken.loc, expr)
    
    def parseNumberExpr(self):
        '''
        Parse a literal number.
        numberexpr ::= number
        '''
        numberToken = self.pop(NumberToken)
        return NumberExprAST(numberToken.loc, numberToken.value)
    
    def parseTensorLiteralExpr(self):
        '''
        Parse a literal array expression.
        tensorLiteral ::= [ literalList ] | number
        literalList ::= tensorLiteral | tensorLiteral, literalList
        '''
        openBracket = self.pop('[')
        
        # Hold the list of values at this nesting level.
        values = []
        # Hold the dimensions for all the nesting inside this level.
        dims = []
        
        while True:
            # We can have either another nested array or a number literal.
            if self.check('['):
                values.append(self.parseTensorLiteralExpr())
            else:
                if not self.check(NumberToken):
                    self.parseError('<num> or [', 'in literal expression')
                values.append(self.parseNumberExpr())
        
            # End of this list on ']'
            if self.check(']'):
                break

            # Elements are separated by a comma.            
            self.pop(',')
        
        self.pop(']')
        
        # Fill in the dimensions now. First the current nesting level:
        dims.append(len(values))
        
        # If there is any nested array, process all of them and ensure 
        # that dimensions are uniform.
        if any(type(val) is LiteralExprAST for val in values):
            first = values[0].dims
            allTensors = all(type(val) is LiteralExprAST for val in values)
            if not allTensors:
                self.parseError('uniform well-nested dimensions', 'inside literal expression')
            
            allEqual = all(val.dims == first for val in values)
            if not allEqual:
                self.parseError('uniform well-nested dimensions', 'inside literal expression')

            dims += first
        
        return LiteralExprAST(openBracket.loc, values, dims)
    
    def parseParenExpr(self):
        "parenexpr ::= '(' expression ')'"
        self.pop('(')
        v = self.parseExpression()
        self.pop(')')
        return v
        
    def parseIdentifierExpr(self):
        '''
        identifierexpr
        ::= identifier
        ::= identifier '(' expression ')'
        '''
        name = self.pop(IdentifierToken)
        if not self.check('('):
            # Simple variable ref.
            return VariableExprAST(name.loc, name.text)
        
        # This is a function call.
        self.pop('(')
        args = []
        while True:
            args.append(self.parseExpression())
            if self.check(')'):
                break
            self.pop(',')
        self.pop(')')
        
        if name.text == 'print':
            # It can be a builtin call to print
            if len(args) != 1:
                self.parseError('<single arg>', 'as argument to print()')
            
            return PrintExprAST(name.loc, args[0])
        
        return CallExprAST(name.loc, name.text, args)
    
    def parsePrimary(self):
        '''
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        '''
        current = self.tokens[self.pos]
        if isinstance(current, IdentifierToken):
            return self.parseIdentifierExpr()
        elif isinstance(current, NumberToken):
            return self.parseNumberExpr()
        elif current.text == '(':
            return self.parseParenExpr()
        elif current.text == '[':
            return self.parseTensorLiteralExpr()
        elif current.text == ';':
            return None
        elif current.text == '}':
            return None
        else:
            raise ParseError(f'Unknown token {current} when expecting an expression')

    def parseBinOpRHS(self, exprPrec, lhs):
        '''
        Recursively parse the right hand side of a binary expression, the ExprPrec
        argument indicates the precedence of the current binary operator.
        
        binoprhs ::= ('+' primary)*
        '''
        # If this is a binop, find its precedence.
        while True:
            tokPrec = self.getTokenPrecedence()
            
            # If this is a binop that binds at least as tightly as the current binop,
            # consume it, otherwise we are done.
            if tokPrec < exprPrec:
                return lhs
            
            # Okay, we know this is a binop.
            binOp = self.pop(OperatorToken).text
            
            # Parse the primary expression after the binary operator.
            rhs = self.parsePrimary()
            
            if rhs is None:
                self.parseError('expression', 'to complete binary operator')
            
            # If BinOp binds less tightly with rhs than the operator after rhs, let
            # the pending operator take rhs as its lhs.
            nextPrec = self.getTokenPrecedence()
            if tokPrec < nextPrec:
                rhs = self.parseBinOpRHS(tokPrec + 1, rhs)

            # Merge lhs/rhs
            lhs = BinaryExprAST(rhs.loc, binOp, lhs, rhs)

    def parseExpression(self):
        'expression::= primary binop rhs'
        lhs = self.parsePrimary()
        return self.parseBinOpRHS(0, lhs)
    
    def parseType(self):
        '''
        type ::= < shape_list >
        shape_list ::= num | num , shape_list
        '''
        self.pop('<')
        shape = []
        
        while self.check(NumberToken):
            shape.append(self.pop().value)
            if self.check('>'):
                self.pop()
                break
            self.pop(',')
        
        return VarType(shape)

    def parseDeclaration(self):
        '''
        Parse a variable declaration, it starts with a `var` keyword followed by
        and identifier and an optional type (shape specification) before the
        initializer.
        decl ::= var identifier [ type ] = expr
        '''
        var = self.pop('var')
        name = self.pop(IdentifierToken).text
        
        # Type is optional, it can be inferred
        if self.check('<'):
            varType = self.parseType()
        else:
            varType = VarType([])
        
        self.pop('=')
        
        expr = self.parseExpression()
        return VarDeclExprAST(var.loc, name, varType, expr)
    
    def parseBlock(self):
        '''
        Parse a block: a list of expression separated by semicolons and wrapped in
        curly braces.

        block ::= { expression_list }
        expression_list ::= block_expr ; expression_list
        block_expr ::= decl | "return" | expr
        '''
        self.pop('{')
        exprList = []
        
        # Ignore empty expressions: swallow sequences of semicolons.
        while self.check(';'):
            self.pop(';')
        
        while not self.check('}'):
            if self.check('var'):
                # Variable declaration
                exprList.append(self.parseDeclaration())
            elif self.check('return'):
                # Return statement
                exprList.append(self.parseReturn())
            else:
                # General expression
                exprList.append(self.parseExpression())
            
            # Ensure that elements are separated by a semicolon.
            self.pop(';')
            
            # Ignore empty expressions: swallow sequences of semicolons.
            while self.check(';'):
                self.pop(';')
        
        self.pop('}')
        
        return exprList
    
    def parsePrototype(self):
        '''
        prototype ::= def id '(' decl_list ')'
        decl_list ::= identifier | identifier, decl_list
        '''
        defToken = self.pop('def')
        fnName = self.pop(IdentifierToken).text
        self.pop('(')
        
        args = []
        if not self.check(')'):
            while True:
                arg = self.pop(IdentifierToken)
                args.append(VariableExprAST(arg.loc, arg.text))
                if not self.check(','):
                    break
                self.pop(',')
        
        self.pop(')')
        return PrototypeAST(defToken.loc, fnName, args)
    
    def parseDefinition(self):
        '''
        Parse a function definition, we expect a prototype initiated with the
        `def` keyword, followed by a block containing a list of expressions.
    
        definition ::= prototype block
        '''
        proto = self.parsePrototype()
        block = self.parseBlock()
        return FunctionAST(proto.loc, proto, block)
    
    def parseError(self, expected, context=''):
        '''
        Helper function to signal errors while parsing, it takes an argument
        indicating the expected token and another argument giving more context.
        Location is retrieved from the lexer to enrich the error message.
        '''
        token = self.getToken()
        line = self.program.splitlines()[token.line]
        raise ParseError(self.getToken(), expected, context, line)

