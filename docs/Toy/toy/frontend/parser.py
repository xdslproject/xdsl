from pathlib import Path
from typing import NoReturn, TypeVar, cast, overload

from .lexer import (
    EOFToken,
    IdentifierToken,
    NumberToken,
    OperatorToken,
    Token,
    tokenize,
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
    VarType,
)


class ParseError(Exception):
    def __init__(
        self,
        token: Token,
        expected: str | type[Token],
        context: str = "",
        line: str = "",
    ):
        loc = token.loc

        message = f"Parse error ({loc.line}, {loc.col}): expected "
        if isinstance(expected, str):
            message += expected
        else:
            message += expected.name()

        if len(context):
            message += " " + context

        message += f" but has Token '{token.text}'\n"

        if len(line):
            message += line + "\n"

        super().__init__(message)

    pass


TokenT = TypeVar("TokenT", bound=Token)


class Parser:
    file: Path
    program: str
    tokens: list[Token]
    pos: int

    def __init__(self, file: Path, program: str):
        self.file = file
        self.program = program
        self.tokens = tokenize(file, program)
        self.pos = 0

    def getToken(self):
        """Returns current token in parser"""
        return self.tokens[self.pos]

    def getTokenPrecedence(self) -> int:
        """Returns precedence if the current token is a binary operation, -1 otherwise"""
        PRECEDENCE = {
            "-": 20,
            "+": 20,
            "*": 40,
        }
        op = self.getToken().text

        return PRECEDENCE.get(op, -1)

    @overload
    def peek(self, pattern: str) -> Token | None:
        ...

    @overload
    def peek(self, pattern: type[TokenT] = Token) -> TokenT | None:
        ...

    def peek(self, pattern: str | type[TokenT] = Token) -> Token | TokenT | None:
        """
        Returns token matching pattern or None
        """
        token = self.getToken()

        if isinstance(pattern, str):
            if token.text == pattern:
                return token
            else:
                return None
        else:
            if isinstance(token, pattern):
                return token
            else:
                return None

    def check(self, pattern: str | type[Token] = Token) -> bool:
        """
        Verifies that the current token fits the pattern,
        returns False otherwise
        """
        return self.peek(pattern) is not None

    def pop(self) -> Token:
        self.pos += 1
        return self.tokens[self.pos - 1]

    def pop_pattern(self, pattern: str) -> Token:
        """
        Verifies that the current token fits the pattern,
        raises ParseError otherwise
        """
        token = self.peek(pattern)
        if token is None:
            self.parseError(f"'{pattern}'")
        self.pos += 1
        return token

    def pop_token(self, tokenType: type[TokenT]) -> TokenT:
        """
        Verifies that the current token is of expected type,
        raises ParseError otherwise
        """
        token = self.peek(tokenType)
        if token is None:
            self.parseError(tokenType)
        self.pos += 1
        return token

    def parseModule(self):
        """
        Parse a full Module. A module is a list of function definitions.
        """
        functions: list[FunctionAST] = []

        while not self.check(EOFToken):
            functions.append(self.parseDefinition())

        # If we didn't reach EOF, there was an error during parsing
        self.pop_token(EOFToken)

        return ModuleAST(tuple(functions))

    def parseReturn(self):
        """
        Parse a return statement.
        return :== return ; | return expr ;
        """
        returnToken = self.pop_pattern("return")
        expr = None

        # Return takes an optional argument
        if not self.check(";"):
            expr = self.parseExpression()

        return ReturnExprAST(returnToken.loc, expr)

    def parseNumberExpr(self):
        """
        Parse a literal number.
        numberexpr ::= number
        """
        numberToken = self.pop_token(NumberToken)
        return NumberExprAST(numberToken.loc, numberToken.value)

    def parseTensorLiteralExpr(self) -> LiteralExprAST | NumberExprAST:
        """
        Parse a literal array expression.
        tensorLiteral ::= [ literalList ] | number
        literalList ::= tensorLiteral | tensorLiteral, literalList
        """
        if self.check(NumberToken):
            return self.parseNumberExpr()

        openBracket = self.pop_pattern("[")

        # Hold the list of values at this nesting level.
        values: list[LiteralExprAST | NumberExprAST] = []
        # Hold the dimensions for all the nesting inside this level.
        dims: list[int] = []

        while True:
            # We can have either another nested array or a number literal.
            if self.check("["):
                values.append(self.parseTensorLiteralExpr())
            else:
                if not self.check(NumberToken):
                    self.parseError("<num> or [", "in literal expression")
                values.append(self.parseNumberExpr())

            # End of this list on ']'
            if self.check("]"):
                break

            # Elements are separated by a comma.
            self.pop_pattern(",")

        self.pop_pattern("]")

        # Fill in the dimensions now. First the current nesting level:
        dims.append(len(values))

        # If there is any nested array, process all of them and ensure
        # that dimensions are uniform.
        if any(type(val) is LiteralExprAST for val in values):
            allTensors = all(type(val) is LiteralExprAST for val in values)
            if not allTensors:
                self.parseError(
                    "uniform well-nested dimensions", "inside literal expression"
                )

            tensor_values = cast(list[LiteralExprAST], values)
            first = tensor_values[0].dims
            allEqual = all(val.dims == first for val in tensor_values)
            if not allEqual:
                self.parseError(
                    "uniform well-nested dimensions", "inside literal expression"
                )

            dims += first

        return LiteralExprAST(openBracket.loc, values, dims)

    def parseParenExpr(self) -> ExprAST:
        "parenexpr ::= '(' expression ')'"
        self.pop_pattern("(")
        v = self.parseExpression()
        self.pop_pattern(")")
        return v

    def parseIdentifierExpr(self):
        """
        identifierexpr
        ::= identifier
        ::= identifier '(' expression ')'
        """
        name = self.pop_token(IdentifierToken)
        if not self.check("("):
            # Simple variable ref.
            return VariableExprAST(name.loc, name.text)

        # This is a function call.
        self.pop_pattern("(")
        args: list[ExprAST] = []
        while True:
            args.append(self.parseExpression())
            if self.check(")"):
                break
            self.pop_pattern(",")
        self.pop_pattern(")")

        if name.text == "print":
            # It can be a builtin call to print
            if len(args) != 1:
                self.parseError("<single arg>", "as argument to print()")

            return PrintExprAST(name.loc, args[0])

        return CallExprAST(name.loc, name.text, args)

    def parsePrimary(self) -> ExprAST | None:
        """
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        """
        current = self.tokens[self.pos]
        if isinstance(current, IdentifierToken):
            return self.parseIdentifierExpr()
        elif isinstance(current, NumberToken):
            return self.parseNumberExpr()
        elif current.text == "(":
            return self.parseParenExpr()
        elif current.text == "[":
            return self.parseTensorLiteralExpr()
        elif current.text == ";":
            return None
        elif current.text == "}":
            return None
        else:
            self.parseError("expression or one of `;`, `}`")

    def parsePrimaryNotNone(self) -> ExprAST:
        """
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        """
        current = self.tokens[self.pos]
        if isinstance(current, IdentifierToken):
            return self.parseIdentifierExpr()
        elif isinstance(current, NumberToken):
            return self.parseNumberExpr()
        elif current.text == "(":
            return self.parseParenExpr()
        elif current.text == "[":
            return self.parseTensorLiteralExpr()
        else:
            self.parseError("expression")

    def parseBinOpRHS(self, exprPrec: int, lhs: ExprAST) -> ExprAST:
        """
        Recursively parse the right hand side of a binary expression, the ExprPrec
        argument indicates the precedence of the current binary operator.

        binoprhs ::= ('+' primary)*
        """
        # If this is a binop, find its precedence.
        while True:
            tokPrec = self.getTokenPrecedence()

            # If this is a binop that binds at least as tightly as the current binop,
            # consume it, otherwise we are done.
            if tokPrec < exprPrec:
                return lhs

            # Okay, we know this is a binop.
            binOp = self.pop_token(OperatorToken).text

            # Parse the primary expression after the binary operator.
            rhs = self.parsePrimary()

            if rhs is None:
                self.parseError("expression", "to complete binary operator")

            # If BinOp binds less tightly with rhs than the operator after rhs, let
            # the pending operator take rhs as its lhs.
            nextPrec = self.getTokenPrecedence()
            if tokPrec < nextPrec:
                rhs = self.parseBinOpRHS(tokPrec + 1, rhs)

            # Merge lhs/rhs
            lhs = BinaryExprAST(rhs.loc, binOp, lhs, rhs)

    def parseExpression(self) -> ExprAST:
        """expression::= primary binop rhs"""
        lhs = self.parsePrimaryNotNone()
        return self.parseBinOpRHS(0, lhs)

    def parseType(self):
        """
        type ::= < shape_list >
        shape_list ::= num | num , shape_list
        """
        self.pop_pattern("<")
        shape: list[int] = []

        while token := self.pop_token(NumberToken):
            shape.append(int(token.value))
            if self.check(">"):
                self.pop()
                break
            self.pop_pattern(",")

        return VarType(shape)

    def parseDeclaration(self):
        """
        Parse a variable declaration, it starts with a `var` keyword followed by
        and identifier and an optional type (shape specification) before the
        initializer.
        decl ::= var identifier [ type ] = expr
        """
        var = self.pop_pattern("var")
        name = self.pop_token(IdentifierToken).text

        # Type is optional, it can be inferred
        if self.check("<"):
            varType = self.parseType()
        else:
            varType = VarType([])

        self.pop_pattern("=")

        expr = self.parseExpression()
        return VarDeclExprAST(var.loc, name, varType, expr)

    def parseBlock(self) -> tuple[ExprAST, ...]:
        """
        Parse a block: a list of expression separated by semicolons and wrapped in
        curly braces.

        block ::= { expression_list }
        expression_list ::= block_expr ; expression_list
        block_expr ::= decl | "return" | expr
        """
        self.pop_pattern("{")
        exprList: list[ExprAST] = []

        # Ignore empty expressions: swallow sequences of semicolons.
        while self.check(";"):
            self.pop_pattern(";")

        while not self.check("}"):
            if self.check("var"):
                # Variable declaration
                exprList.append(self.parseDeclaration())
            elif self.check("return"):
                # Return statement
                exprList.append(self.parseReturn())
            else:
                # General expression
                exprList.append(self.parseExpression())

            # Ensure that elements are separated by a semicolon.
            self.pop_pattern(";")

            # Ignore empty expressions: swallow sequences of semicolons.
            while self.check(";"):
                self.pop_pattern(";")

        self.pop_pattern("}")

        return tuple(exprList)

    def parsePrototype(self):
        """
        prototype ::= def id '(' decl_list ')'
        decl_list ::= identifier | identifier, decl_list
        """
        defToken = self.pop_pattern("def")
        fnName = self.pop_token(IdentifierToken).text
        self.pop_pattern("(")

        args: list[VariableExprAST] = []
        if not self.check(")"):
            while True:
                arg = self.pop_token(IdentifierToken)
                args.append(VariableExprAST(arg.loc, arg.text))
                if not self.check(","):
                    break
                self.pop_pattern(",")

        self.pop_pattern(")")
        return PrototypeAST(defToken.loc, fnName, args)

    def parseDefinition(self):
        """
        Parse a function definition, we expect a prototype initiated with the
        `def` keyword, followed by a block containing a list of expressions.

        definition ::= prototype block
        """
        proto = self.parsePrototype()
        block = self.parseBlock()
        return FunctionAST(proto.loc, proto, block)

    def parseError(self, expected: str | type[Token], context: str = "") -> NoReturn:
        """
        Helper function to signal errors while parsing, it takes an argument
        indicating the expected token and another argument giving more context.
        Location is retrieved from the lexer to enrich the error message.
        """
        token = self.getToken()
        line = self.program.splitlines()[token.line - 1]
        raise ParseError(self.getToken(), expected, context, line)
