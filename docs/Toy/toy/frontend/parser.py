from pathlib import Path
from typing import cast

from xdsl.parser import GenericParser, ParserState
from xdsl.utils.lexer import Input

from .lexer import ToyLexer, ToyToken, ToyTokenKind
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

    def get_token_precedence(self) -> int:
        """Returns precedence if the current token is a binary operation, -1 otherwise"""
        PRECEDENCE = {
            "-": 20,
            "+": 20,
            "*": 40,
        }
        op = self._current_token.text

        return PRECEDENCE.get(op, -1)

    def _peek(self, expected_kind: ToyTokenKind) -> ToyToken | None:
        """
        Returns token matching pattern or None.
        """
        token = self._current_token
        if token.kind == expected_kind:
            return token

    def _pop(self, expected_kind: ToyTokenKind) -> ToyToken:
        """
        Verifies that the current token is of expected type,
        raises ParseError otherwise.
        """
        return self._parse_token(expected_kind, f"Expected {expected_kind}")

    def parse_module(self):
        """
        Parse a full Module. A module is a list of function definitions.
        """
        functions: list[FunctionAST] = []

        while self._parse_optional_token(ToyTokenKind.EOF) is None:
            functions.append(self.parse_definition())

        return ModuleAST(tuple(functions))

    def parse_return(self):
        """
        Parse a return statement.
        return :== return ; | return expr ;
        """
        return_token = self._pop(ToyTokenKind.RETURN)
        expr = None

        # Return takes an optional argument
        if self._parse_optional_token(ToyTokenKind.SEMICOLON) is None:
            expr = self.parse_expression()

        return ReturnExprAST(return_token.span.get_location(), expr)

    def parse_number_expr(self):
        """
        Parse a literal number.
        numberexpr ::= number
        """
        number_token = self._pop(ToyTokenKind.NUMBER)
        return NumberExprAST(
            number_token.span.get_location(), float(number_token.span.text)
        )

    def parse_tensor_literal_expr(self) -> LiteralExprAST | NumberExprAST:
        """
        Parse a literal array expression.
        tensorLiteral ::= [ literalList ] | number
        literalList ::= tensorLiteral | tensorLiteral, literalList
        """
        if self._peek(ToyTokenKind.NUMBER):
            return self.parse_number_expr()

        open_bracket = self._current_token

        # Hold the list of values at this nesting level.
        values = self.parse_comma_separated_list(
            self.Delimiter.SQUARE, self.parse_tensor_literal_expr
        )

        # Hold the dimensions for all the nesting inside this level.
        # Fill in the dimensions now. First the current nesting level:
        dims = [len(values)]

        # If there is any nested array, process all of them and ensure
        # that dimensions are uniform.
        if any(type(val) is LiteralExprAST for val in values):
            all_tensors = all(type(val) is LiteralExprAST for val in values)
            if not all_tensors:
                self.raise_error(
                    "Expected uniform well-nested dimensions inside literal expression"
                )

            tensor_values = cast(list[LiteralExprAST], values)
            first = tensor_values[0].dims
            all_equal = all(val.dims == first for val in tensor_values)
            if not all_equal:
                self.raise_error(
                    "Expected uniform well-nested dimensions inside literal expression"
                )

            dims += first

        return LiteralExprAST(open_bracket.span.get_location(), values, dims)

    def parse_paren_expr(self) -> ExprAST:
        "parenexpr ::= '(' expression ')'"
        with self.in_parens():
            return self.parse_expression()

    def parse_identifier_expr(self):
        """
        identifierexpr
        ::= identifier
        ::= identifier '(' expression ')'
        """
        name = self._pop(ToyTokenKind.IDENTIFIER)
        args = self.parse_optional_comma_separated_list(
            self.Delimiter.PAREN, self.parse_expression
        )
        if args is None:
            # Simple variable ref.
            return VariableExprAST(name.span.get_location(), name.text)

        # This is a function call.
        if name.text == "print":
            # It can be a builtin call to print
            if len(args) != 1:
                self.raise_error("Expected <single arg> as argument to print()")

            return PrintExprAST(name.span.get_location(), args[0])

        return CallExprAST(name.span.get_location(), name.text, args)

    def parse_primary(self) -> ExprAST | None:
        """
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        """
        current = self._current_token
        if current.kind == ToyTokenKind.IDENTIFIER:
            return self.parse_identifier_expr()
        elif current.kind == ToyTokenKind.NUMBER:
            return self.parse_number_expr()
        elif current.kind == ToyTokenKind.PARENTHESE_OPEN:
            return self.parse_paren_expr()
        elif current.kind == ToyTokenKind.SBRACKET_OPEN:
            return self.parse_tensor_literal_expr()
        elif current.kind == ToyTokenKind.SEMICOLON:
            return None
        elif current.kind == ToyTokenKind.BRACKET_CLOSE:
            return None
        else:
            self.raise_error("Expected expression or one of `;`, `}`")

    def parse_primary_not_none(self) -> ExprAST:
        """
        primary
        ::= identifierexpr
        ::= numberexpr
        ::= parenexpr
        ::= tensorliteral
        """
        current = self._current_token
        if current.kind == ToyTokenKind.IDENTIFIER:
            return self.parse_identifier_expr()
        elif current.kind == ToyTokenKind.NUMBER:
            return self.parse_number_expr()
        elif current.kind == ToyTokenKind.PARENTHESE_OPEN:
            return self.parse_paren_expr()
        elif current.kind == ToyTokenKind.SBRACKET_OPEN:
            return self.parse_tensor_literal_expr()
        else:
            self.raise_error("Expected expression")

    def parse_bin_op_rhs(self, expr_precedence: int, lhs: ExprAST) -> ExprAST:
        """
        Recursively parse the right hand side of a binary expression, the ExprPrec
        argument indicates the precedence of the current binary operator.

        binoprhs ::= ('+' primary)*
        """
        # If this is a binop, find its precedence.
        while True:
            tok_precedence = self.get_token_precedence()

            # If this is a binop that binds at least as tightly as the current binop,
            # consume it, otherwise we are done.
            if tok_precedence < expr_precedence:
                return lhs

            # Okay, we know this is a binop.
            binOp = self._pop(ToyTokenKind.OPERATOR).text

            # Parse the primary expression after the binary operator.
            rhs = self.parse_primary()

            if rhs is None:
                self.raise_error("Expected expression to complete binary operator")

            # If BinOp binds less tightly with rhs than the operator after rhs, let
            # the pending operator take rhs as its lhs.
            next_precedence = self.get_token_precedence()
            if tok_precedence < next_precedence:
                rhs = self.parse_bin_op_rhs(tok_precedence + 1, rhs)

            # Merge lhs/rhs
            lhs = BinaryExprAST(rhs.loc, binOp, lhs, rhs)

    def parse_expression(self) -> ExprAST:
        """expression::= primary binop rhs"""
        lhs = self.parse_primary_not_none()
        return self.parse_bin_op_rhs(0, lhs)

    def parse_type(self):
        """
        type ::= < shape_list >
        shape_list ::= num | num , shape_list
        """
        return VarType(
            self.parse_comma_separated_list(
                self.Delimiter.ANGLE,
                lambda: int(self._pop(ToyTokenKind.NUMBER).text),
            )
        )

    def parse_declaration(self):
        """
        Parse a variable declaration, it starts with a `var` keyword followed by
        and identifier and an optional type (shape specification) before the
        initializer.
        decl ::= var identifier [ type ] = expr
        """
        var = self._pop(ToyTokenKind.VAR)
        name = self._pop(ToyTokenKind.IDENTIFIER).text
        var_type = self.parse_type() if self._peek(ToyTokenKind.LT) else VarType([])
        self._pop(ToyTokenKind.EQ)
        expr = self.parse_expression()
        return VarDeclExprAST(var.span.get_location(), name, var_type, expr)

    def parse_block(self) -> tuple[ExprAST, ...]:
        """
        Parse a block: a list of expression separated by semicolons and wrapped in
        curly braces.

        block ::= { expression_list }
        expression_list ::= block_expr ; expression_list
        block_expr ::= decl | "return" | expr
        """
        self._pop(ToyTokenKind.BRACKET_OPEN)
        exprs: list[ExprAST] = []

        # Ignore empty expressions: swallow sequences of semicolons.
        while self._parse_optional_token(ToyTokenKind.SEMICOLON):
            continue

        while not self._parse_optional_token(ToyTokenKind.BRACKET_CLOSE):
            if self._peek(ToyTokenKind.VAR):
                # Variable declaration
                exprs.append(self.parse_declaration())
            elif self._peek(ToyTokenKind.RETURN):
                # Return statement
                exprs.append(self.parse_return())
            else:
                # General expression
                exprs.append(self.parse_expression())

            # Ensure that elements are separated by a semicolon.
            self._pop(ToyTokenKind.SEMICOLON)

            # Ignore empty expressions: swallow sequences of semicolons.
            while self._parse_optional_token(ToyTokenKind.SEMICOLON):
                continue

        return tuple(exprs)

    def _parse_arg(self) -> VariableExprAST:
        arg = self._pop(ToyTokenKind.IDENTIFIER)
        return VariableExprAST(arg.span.get_location(), arg.text)

    def parse_prototype(self):
        """
        prototype ::= def id '(' decl_list ')'
        decl_list ::= identifier | identifier, decl_list
        """
        def_token = self._pop(ToyTokenKind.DEF)
        name = self._pop(ToyTokenKind.IDENTIFIER).text
        args = self.parse_comma_separated_list(self.Delimiter.PAREN, self._parse_arg)
        return PrototypeAST(def_token.span.get_location(), name, args)

    def parse_definition(self):
        """
        Parse a function definition, we expect a prototype initiated with the
        `def` keyword, followed by a block containing a list of expressions.

        definition ::= prototype block
        """
        proto = self.parse_prototype()
        block = self.parse_block()
        return FunctionAST(proto.loc, proto, block)
