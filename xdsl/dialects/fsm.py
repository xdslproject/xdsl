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
from xdsl.ir.core import Dialect, ParametrizedAttribute, TypeAttribute
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
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.irdl.irdl import irdl_attr_definition
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoTerminator,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))


@irdl_attr_definition
class InstanceType(ParametrizedAttribute, TypeAttribute):
    name = "fsm.instancetype"


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

    traits = frozenset([NoTerminator()])

    def __init__(
        self,
        nextState: FlatSymbolRefAttr,
        guard: Region | type[Region.DEFAULT] = Region.DEFAULT,
        action: Region | type[Region.DEFAULT] = Region.DEFAULT,
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
        if SymbolTable.lookup_symbol(self, self.nextState) is None or not isinstance(
            SymbolTable.lookup_symbol(self, self.nextState), State
        ):
            raise VerifyException("Can not find next state")
        if self.guard.blocks and not isinstance(self.guard.block.last_op, Return):
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

        super().__init__(attributes=attributes, regions=[body])

    def verify_(self):
        if not SymbolTable.lookup_symbol(self, self.initialState):
            raise VerifyException("Can not find initial state")
        if (
            self.arg_attrs is not None
            and self.arg_names is None
            or self.res_attrs is not None
            and self.res_names is None
            or self.arg_attrs is None
            and self.arg_names is not None
            or self.res_attrs is None
            and self.res_names is not None
        ):
            raise VerifyException("attrs must be consistent with names")
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
        if (
            isinstance(parent, State)
            and parent.transitions == self.parent_region()
            and len(self.operands) > 0
        ):
            raise VerifyException("Transition regions should not output any value")
        while parent is not None:
            if isinstance(parent, Machine):
                if not (
                    [operand.type for operand in self.operands]
                    == [result for result in parent.function_type.outputs]
                    and len(self.operands) == len(parent.function_type.outputs)
                ):
                    raise VerifyException(
                        "Output type must be consistent with the machine's"
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

    traits = frozenset([HasParent(Transition)])

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


@irdl_op_definition
class Instance(IRDLOperation):
    """Represents an instance of a state machine, including an
    instance name and a symbol reference of the machine"""

    name = "fsm.instance"

    sym_name = attr_def(StringAttr)

    machine = attr_def(FlatSymbolRefAttr)

    res = result_def(InstanceType)

    def __init__(
        self, sym_name: str, machine: FlatSymbolRefAttr, instance: InstanceType
    ):
        if isinstance(machine, str):
            machine = FlatSymbolRefAttr(machine)
        attributes: dict[str, Attribute] = {}
        attributes["sym_name"] = StringAttr(sym_name)
        attributes["machine"] = machine
        super().__init__(result_types=[instance], attributes=attributes)

    def verify_(self):
        if not isinstance(SymbolTable.lookup_symbol(self, self.machine), Machine):
            raise VerifyException("Machine definition does not exist")

    def getMachine(self) -> Machine | None:
        m = SymbolTable.lookup_symbol(self, self.machine)
        if isinstance(m, Machine):
            return m
        else:
            return None


@irdl_op_definition
class Trigger(IRDLOperation):
    """Triggers a state machine instance. The inputs and outputs are
    correponding to the inputs and outputs of the referenced machine of the
    instance"""

    name = "fsm.trigger"

    # operands

    inputs = var_operand_def(AnyAttr())

    instance = operand_def(InstanceType)

    outputs = var_result_def(AnyAttr())

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        instance: SSAValue,
        outputs: Sequence[Attribute],
    ):
        super().__init__(
            operands=[inputs, instance],
            result_types=[outputs],
        )

    def verify_(self):
        if not isinstance(self.instance.owner, Instance):
            raise VerifyException("The instance operand must be Instance")

        # operand types must match the machine input types

        # result types must match the machine output types
        m = self.instance.owner.getMachine()

        if m is None:
            raise VerifyException("Machine definition does not exist.")

        if not (
            [operand.type for operand in self.operands]
            == [type(result) for result in m.function_type.outputs]
            and len(self.operands) == len(m.function_type.outputs)
        ):
            raise VerifyException("Output type must be consistent with the machine's")

        if not (
            [operand.type for operand in self.inputs]
            == [type(result) for result in m.function_type.inputs]
            and len(self.operands) == len(m.function_type.inputs)
        ):
            raise VerifyException("Input types must be consistent with the machine's")


@irdl_op_definition
class HWInstance(IRDLOperation):
    """Represents a hardware-style instance of a state machine,
    including an instance name and a symbol reference of the machine. The inputs
    and outputs are correponding to the inputs and outputs of the referenced
    machine."""

    name = "fsm.hw_instance"

    sym_name = attr_def(StringAttr)
    machine = attr_def(FlatSymbolRefAttr)
    inputs = var_operand_def(AnyAttr())
    clock = operand_def(signlessIntegerLike)
    reset = operand_def(signlessIntegerLike)

    outputs = var_result_def(AnyAttr())

    def __init__(
        self,
        sym_name: str | StringAttr,
        machine: str | FlatSymbolRefAttr,
        inputs: Sequence[SSAValue | Operation],
        clock: SSAValue | Operation,
        reset: SSAValue | Operation,
        outputs: Sequence[Attribute],
    ):
        if isinstance(machine, str):
            machine = FlatSymbolRefAttr(machine)
        clock = SSAValue.get(clock)
        reset = SSAValue.get(reset)
        attributes: dict[str, Attribute] = {}
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        attributes["sym_name"] = sym_name
        attributes["machine"] = machine
        super().__init__(
            operands=[inputs, clock, reset],
            result_types=[outputs],
            attributes=attributes,
        )

    def verify_(self):
        m = SymbolTable.lookup_symbol(self.parent_op(), self.machine)
        print("machine is " + str(self.machine))
        if isinstance(m, Machine):
            if not (
                [operand.type for operand in self.inputs]
                == [type(result) for result in m.function_type.outputs]
                and len(self.operands) == len(m.function_type.outputs)
            ):
                raise VerifyException(
                    "Output types must be consistent with the machine's"
                )
            if not (
                [operand.type for operand in self.inputs]
                == [type(result) for result in m.function_type.inputs]
                and len(self.operands) == len(m.function_type.inputs)
            ):
                raise VerifyException(
                    "Input types must be consistent with the machine's"
                )
        else:
            raise VerifyException(
                "Machine definition does not exist and it is "
                + str(SymbolTable.lookup_symbol(self, self.machine))
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
        Trigger,
        Instance,
        HWInstance,
    ],
    [InstanceType],
)
