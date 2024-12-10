from pathlib import Path
from typing import cast

from xdsl.parser import GenericParser, ParserState
from xdsl.utils.lexer import Input

from .lexer import ToyLexer, ToyToken, ToyTokenKind
from .location import loc
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


class ToyParser(GenericParser[ToyTokenKind]):
    def __init__(self, file: Path, program: str):
        super().__init__(ParserState(ToyLexer(Input(program, str(file)))))

    def getToken(self):
        """Returns current token in parser"""
        return self._current_token

    def getTokenPrecedence(self) -> int:
        """Returns precedence if the current token is a binary operation, -1 otherwise"""
        PRECEDENCE = {
            "-": 20,
            "+": 20,
            "*": 40,
        }
        op = self._current_token.text

        return PRECEDENCE.get(op, -1)

    def peek(self, pattern: str | ToyTokenKind) -> ToyToken | None:
        """
        Returns token matching pattern or None
        """
        token = self._current_token

        if isinstance(pattern, str):
            if token.text == pattern:
                return token
        else:
            if token.kind == pattern:
                return token

    def check(self, pattern: str | ToyTokenKind) -> bool:
        """
        Verifies that the current token fits the pattern,
        returns False otherwise
        """
        return self.peek(pattern) is not None

    def pop(self) -> ToyToken:
        return self._consume_token()

    def pop_pattern(self, pattern: str) -> ToyToken:
        """
        Verifies that the current token fits the pattern,
        raises ParseError otherwise
        """
        token = self._consume_token()
        if token.text != pattern:
            self.raise_error(f"Expected '{pattern}'", token.span.start, token.span.end)
        return token

    def pop_token(self, tokenType: ToyTokenKind) -> ToyToken:
        """
        Verifies that the current token is of expected type,
        raises ParseError otherwise
        """
        return self._consume_token(tokenType)

    def parseModule(self):
        """
        Parse a full Module. A module is a list of function definitions.
        """
        functions: list[FunctionAST] = []

        while not self.check(ToyTokenKind.EOF):
            functions.append(self.parseDefinition())

        # If we didn't reach EOF, there was an error during parsing
        self.pop_token(ToyTokenKind.EOF)

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

        return ReturnExprAST(loc(returnToken), expr)

    def parseNumberExpr(self):
        """
        Parse a literal number.
        numberexpr ::= number
        """
        numberToken = self.pop_token(ToyTokenKind.NUMBER)
        return NumberExprAST(loc(numberToken), float(numberToken.span.text))

    def parseTensorLiteralExpr(self) -> LiteralExprAST | NumberExprAST:
        """
        Parse a literal array expression.
        tensorLiteral ::= [ literalList ] | number
        literalList ::= tensorLiteral | tensorLiteral, literalList
        """
        if self.check(ToyTokenKind.NUMBER):
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
                if not self.check(ToyTokenKind.NUMBER):
                    self.raise_error("Expected <num> or [ in literal expression")
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
                self.raise_error(
                    "Expected uniform well-nested dimensions inside literal expression"
                )

            tensor_values = cast(list[LiteralExprAST], values)
            first = tensor_values[0].dims
            allEqual = all(val.dims == first for val in tensor_values)
            if not allEqual:
                self.raise_error(
                    "Expected uniform well-nested dimensions inside literal expression"
                )

            dims += first

        return LiteralExprAST(loc(openBracket), values, dims)

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
        name = self.pop_token(ToyTokenKind.IDENTIFIER)
        if not self.check("("):
            # Simple variable ref.
            return VariableExprAST(loc(name), name.text)

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
                self.raise_error("Expected <single arg> as argument to print()")

            return PrintExprAST(loc(name), args[0])

        return CallExprAST(loc(name), name.text, args)

    def parsePrimary(self) -> ExprAST | None:
        """
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        """
        current = self._current_token
        if current.kind == ToyTokenKind.IDENTIFIER:
            return self.parseIdentifierExpr()
        elif current.kind == ToyTokenKind.NUMBER:
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
            self.raise_error("Expected expression or one of `;`, `}`")

    def parsePrimaryNotNone(self) -> ExprAST:
        """
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        """
        current = self._current_token
        if current.kind == ToyTokenKind.IDENTIFIER:
            return self.parseIdentifierExpr()
        elif current.kind == ToyTokenKind.NUMBER:
            return self.parseNumberExpr()
        elif current.text == "(":
            return self.parseParenExpr()
        elif current.text == "[":
            return self.parseTensorLiteralExpr()
        else:
            self.raise_error("Expected expression")

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
            binOp = self.pop_token(ToyTokenKind.OPERATOR).text

            # Parse the primary expression after the binary operator.
            rhs = self.parsePrimary()

            if rhs is None:
                self.raise_error("Expected expression to complete binary operator")

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

        while token := self.pop_token(ToyTokenKind.NUMBER):
            shape.append(int(token.span.text))
            if self.parse_optional_characters(">"):
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
        name = self.pop_token(ToyTokenKind.IDENTIFIER).text

        # Type is optional, it can be inferred
        if self.check("<"):
            varType = self.parseType()
        else:
            varType = VarType([])

        self.pop_pattern("=")

        expr = self.parseExpression()
        return VarDeclExprAST(loc(var), name, varType, expr)

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
        fnName = self.pop_token(ToyTokenKind.IDENTIFIER).text
        self.pop_pattern("(")

        args: list[VariableExprAST] = []
        if not self.check(")"):
            while True:
                arg = self.pop_token(ToyTokenKind.IDENTIFIER)
                args.append(VariableExprAST(loc(arg), arg.text))
                if not self.check(","):
                    break
                self.pop_pattern(",")

        self.pop_pattern(")")
        return PrototypeAST(loc(defToken), fnName, args)

    def parseDefinition(self):
        """
        Parse a function definition, we expect a prototype initiated with the
        `def` keyword, followed by a block containing a list of expressions.

        definition ::= prototype block
        """
        proto = self.parsePrototype()
        block = self.parseBlock()
        return FunctionAST(proto.loc, proto, block)
