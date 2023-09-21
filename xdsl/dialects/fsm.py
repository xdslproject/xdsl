from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerOf,
    DictionaryAttr,
    FlatSymbolRefAttr,
    FunctionType,
    IndexType,
    IntegerType,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
    Operation,
    Region,
    SSAValue,
)
from xdsl.ir.core import Dialect
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoTerminator,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))


@irdl_op_definition
class Transition(IRDLOperation):
    """Represents a transition of a state with a symbol reference
    of the next state. This op includes an optional `$guard` region with an `fsm.return`
    as terminator that returns a Boolean value indicating the guard condition of
    this transition. This op also includes an optional `$action` region that represents
    the actions to be executed when this transition is taken"""

    name = "fsm.transition"

    guard = region_def()

    action = region_def()

    # attributes

    nextState = attr_def(FlatSymbolRefAttr)

    def __init__(
        self,
        nextState: FlatSymbolRefAttr,
        action: Region | type[Region.DEFAULT] = Region.DEFAULT,
        guard: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        if isinstance(nextState, str):
            nextState = FlatSymbolRefAttr(nextState)
        attributes: dict[str, Attribute] = {}
        attributes["nextState"] = nextState
        if not isinstance(action, Region):
            action = Region(Block())
        if not isinstance(guard, Region):
            guard = Region(Block())
        super().__init__(
            attributes=attributes,
            regions=[guard, action],
        )

    def verify_(self):
        var = SymbolTable.lookup_symbol(self, self.nextState)
        if var is None:
            raise VerifyException("1. Can not find next state")
        if not isinstance(var, State):
            raise VerifyException("2. Can not find next state")
        if (
            self.guard.blocks
            and self.guard.block.first_op is not None
            and not isinstance(self.guard.block.last_op, Return)
        ):
            raise VerifyException("Guard region must terminate with ReturnOp")
        var = self.parent_op()
        assert isinstance(var, State)
        if var.transitions != self.parent_region():
            raise VerifyException("Transition must be located in a transitions region")


@irdl_op_definition
class Machine(IRDLOperation):
    """Represents a finite-state machine, including a machine name,
    the type of machine state, and the types of inputs and outputs. This op also
    includes a `$body` region that contains internal variables and states."""

    name = "fsm.machine"

    body: Region = region_def()

    sym_name = attr_def(StringAttr)
    initialState = attr_def(StringAttr)
    function_type = attr_def(FunctionType)
    arg_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    arg_names = opt_attr_def(ArrayAttr[StringAttr])
    res_names = opt_attr_def(ArrayAttr[StringAttr])

    traits = frozenset([NoTerminator(), SymbolTable()])

    def __init__(
        self,
        sym_name: str,
        initial_state: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        arg_attrs: ArrayAttr[DictionaryAttr] | None,
        res_attrs: ArrayAttr[DictionaryAttr] | None,
        arg_names: ArrayAttr[StringAttr] | None,
        res_names: ArrayAttr[StringAttr] | None,
        body: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        attributes: dict[str, Attribute | None] = {}
        attributes["sym_name"] = StringAttr(sym_name)
        attributes["initialState"] = StringAttr(initial_state)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        attributes["function_type"] = function_type
        attributes["arg_attrs"] = arg_attrs
        attributes["res_attrs"] = res_attrs
        attributes["arg_names"] = arg_names
        attributes["res_names"] = res_names
        if not isinstance(body, Region):
            body = Region(Block())

        super().__init__(attributes=attributes)

    def verify_(self):
        if not SymbolTable.lookup_symbol(self, self.initialState):
            raise VerifyException("Can not find initial state")
        if self.arg_attrs is not None and self.arg_names is None:
            raise VerifyException("arg_attrs must be consistent with arg_names")
        if self.res_attrs is not None and self.res_names is None:
            raise VerifyException("res_attrs must be consistent with res_names")
        if self.arg_attrs is not None and self.arg_names is not None:
            if len(self.arg_attrs) != len(self.arg_names):
                raise VerifyException(
                    "The number of arg_attrs and arg_names should be the same"
                )
        if self.res_attrs is not None and self.res_names is not None:
            if len(self.res_attrs) != len(self.res_names):
                raise VerifyException(
                    "The number of res_attrs and res_names should be the same"
                )


@irdl_op_definition
class Output(IRDLOperation):
    """Represents the outputs of a machine under a specific state. The
    types of `$operands` should be consistent with the output types of the state
    machine"""

    name = "fsm.output"

    operand = var_operand_def(AnyAttr())

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        operand: Sequence[SSAValue | Operation],
    ):
        super().__init__(
            operands=[operand],
        )

    def verify_(self):
        parent = self.parent_op()
        while parent is not None:
            if isinstance(parent, Transition) and len(self.operands) > 0:
                raise VerifyException("Transition regions should not output any value")
            elif isinstance(parent, Machine):
                # check that the type of the operand
                # is the same as at least one in the type of the entry
                if isinstance(self.operands, type(getattr(parent, "res_attrs"))):
                    raise VerifyException(
                        "Output types must be consistent with the machine"
                    )
            parent = parent.parent_op()


@irdl_op_definition
class State(IRDLOperation):
    """Represents a state of a state machine. This op includes an
    `$output` region with an `fsm.output` as terminator to define the machine
    outputs under this state. This op also includes a `transitions` region that
    contains all the transitions of this state"""

    name = "fsm.state"

    # includes output and transitions region

    output = region_def()

    transitions = region_def()

    # attributes

    sym_name = attr_def(StringAttr)

    traits = frozenset([NoTerminator(), SymbolOpInterface()])

    def __init__(
        self,
        sym_name: str,
        output: Region | type[Region.DEFAULT] = Region.DEFAULT,
        transitions: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        attributes: dict[str, Attribute] = {}
        attributes["sym_name"] = StringAttr(sym_name)
        if not isinstance(output, Region):
            output = Region(Block())
        if not isinstance(transitions, Region):
            transitions = Region(Block())
        super().__init__(
            attributes=attributes,
            regions=[output, transitions],
        )

    def verify_(self):
        parent = self.parent_op()

        while parent is not None:
            if (
                isinstance(parent, Machine)
                and getattr(parent, "res_attrs") is not None
                and len(getattr(parent, "res_attrs")) > 0
                and self.output.block.first_op is None
            ):
                raise VerifyException(
                    "State must have a non-empty output region when the machine has results."
                )
            parent = parent.parent_op()


@irdl_op_definition
class Update(IRDLOperation):
    """Updates the `$variable` with the `$value`. The definition op of
    `$variable` should be an `fsm.variable`. This op should *only* appear in the
    `action` region of a transtion"""

    name = "fsm.update"

    # operands

    variable = operand_def(Attribute)

    value = operand_def(Attribute)

    traits = frozenset([IsTerminator()])

    def __init__(
        self,
        variable: SSAValue | Operation,
        value: SSAValue | Operation,
    ):
        super().__init__(
            operands=[variable, value],
        )

    def verify_(self) -> None:
        if not isinstance(self.variable.owner, Variable):
            raise VerifyException("Destination is not a variable operation")

        parent = self.parent_op()
        while parent is not None:
            if isinstance(parent, Transition):
                # walk through the action region
                found = 0
                for op in parent.action.walk():
                    if isinstance(op, Update) and op.variable == self.variable:
                        found += 1
                if found == 0:
                    raise VerifyException(
                        "Update must only be located in the action region of a transition"
                    )
                elif found > 1:
                    raise VerifyException(
                        "Multiple updates to the same variable within a single action region is disallowed"
                    )
            parent = parent.parent_op()


@irdl_op_definition
class Variable(IRDLOperation):
    """Represents an internal variable in a state machine with an
    initialization value"""

    name = "fsm.variable"

    # attributes

    initValue = attr_def(Attribute)
    name_var = opt_attr_def(StringAttr)

    # results

    result = var_result_def(Attribute)

    def __init__(
        self,
        initValue: Attribute,
        name_var: str | None,
        result: Sequence[Attribute],
    ):
        attributes: dict[str, Attribute] = {}
        attributes["initValue"] = initValue
        if name_var is not None:
            attributes["name_var"] = StringAttr(name_var)
        super().__init__(
            result_types=[result],
            attributes=attributes,
        )


@irdl_op_definition
class Return(IRDLOperation):
    """Marks the end of a region of `fsm.transition` and return
    values if the parent region is a `$guard` region"""

    name = "fsm.return"

    operand = opt_operand_def(signlessIntegerLike)

    traits = frozenset([IsTerminator(), HasParent(Transition)])

    def __init__(
        self,
        operand: SSAValue | Operation,
    ):
        super().__init__(
            operands=[operand],
        )


FSM = Dialect(
    [
        Machine,
        Output,
        State,
        Transition,
        Update,
        Variable,
        Return,
    ],
    [],
)
