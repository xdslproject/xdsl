from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerOf,
    DictionaryAttr,
    FlatSymbolRefAttr,
    FunctionType,
    IndexType,
    InstanceType,
    IntegerType,
    StringAttr,
)
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_region_def,
    region_def,
    result_def,
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
        if SymbolTable.lookup_symbol(self, self.machine) is None:
            raise VerifyException("The machine does not exist")
        if getattr(SymbolOpInterface(self.machine), "arg_names") is not None:
            if isinstance(
                self.inputs, type(getattr(SymbolOpInterface(self.machine), "arg_names"))
            ):
                raise VerifyException("Input is not consistent with machine inputs")
        if getattr(SymbolOpInterface(self.machine), "res_names") is not None:
            if isinstance(
                self.outputs,
                type(getattr(SymbolOpInterface(self.machine), "res_names")),
            ):
                raise VerifyException("Output is not consistent with machine outputs")


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
        assert isinstance(self.res.type, InstanceType)
        if SymbolTable.lookup_symbol(self, self.machine) is None:
            raise VerifyException("The machine does not exist")


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
        attributes: dict[str, Attribute] = {}
        attributes["sym_name"] = StringAttr(sym_name)
        attributes["initialState"] = StringAttr(initial_state)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        attributes["function_type"] = function_type
        if arg_attrs is not None:
            arg_attrs = arg_attrs
            attributes["arg_attrs"] = arg_attrs
        if res_attrs is not None:
            res_attrs = res_attrs
            attributes["res_attrs"] = res_attrs
        if arg_names is not None:
            arg_names = arg_names
            attributes["arg_names"] = arg_names
        if res_names is not None:
            res_names = res_names
            attributes["res_names"] = res_names
        if not isinstance(body, Region):
            body = Region(Block())

        super().__init__(attributes=attributes)

    def verify_(self):
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

    def __init__(
        self,
        operand: Sequence[SSAValue | Operation],
    ):
        super().__init__(
            operands=[operand],
        )

    def verify_(self):
        parent = self.parent_op()
        res_type_machine = type(None)
        while HasParent(Machine):
            if isinstance(parent, Machine):
                res_type_machine = type(parent.arg_attrs)
            if parent is not None:
                parent = parent.parent_op()
            else:
                raise VerifyException("Output must be in a machine")
            if res_type_machine is not type(None):
                if not isinstance(res_type_machine, type(self.operands)):
                    raise VerifyException(
                        "Output type must be consistent with the machine"
                    )


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
        if NoTerminator(self.output.parent_region()):
            raise VerifyException("Output region must have a terminator")


@irdl_op_definition
class Transition(IRDLOperation):
    """Represents a transition of a state with a symbol reference
    of the next state. This op includes an optional `$guard` region with an `fsm.return`
    as terminator that returns a Boolean value indicating the guard condition of
    this transition. This op also includes an optional `$action` region that represents
    the actions to be executed when this transition is taken"""

    name = "fsm.transition"

    guard = opt_region_def()

    action = opt_region_def()

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
        if self.guard is not None and NoTerminator(self.guard):
            raise VerifyException("Guard region must have a terminator")


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
        instance: None,
        outputs: Sequence[Attribute],
    ):
        super().__init__(
            operands=[inputs, instance],
            result_types=[outputs],
        )

    def verify_(self):
        if not (SymbolOpInterface(self.instance) == Instance):
            raise VerifyException("The instance operand must be Instance")
        if not isinstance(
            getattr(SymbolOpInterface(self.instance), "arg_names"), type(self.inputs)
        ):
            raise VerifyException("Input is not consistent with machine inputs")
        if not isinstance(
            getattr(SymbolOpInterface(self.instance), "res_names"), type(self.outputs)
        ):
            raise VerifyException("Output is not consistent with machine outputs")


@irdl_op_definition
class Update(IRDLOperation):
    """Updates the `$variable` with the `$value`. The definition op of
    `$variable` should be an `fsm.variable`. This op should *only* appear in the
    `action` region of a transtion"""

    name = "fsm.update"

    # operands

    variable = operand_def(Attribute)

    value = operand_def(Attribute)

    def __init__(
        self,
        variable: SSAValue | Operation,
        value: SSAValue | Operation,
    ):
        super().__init__(
            operands=[variable, value],
        )

    def verify_(self):
        if not (SymbolOpInterface(self.variable) == Variable):
            raise VerifyException(
                "The definition operator of variable attribute must be Vatiable"
            )

        if not (HasParent(Transition)):
            raise VerifyException(
                "Update should only appear in the action region of a transition"
            )


@irdl_op_definition
class Variable(IRDLOperation):
    """Represents an internal variable in a state machine with an
    initialization value"""

    name = "fsm.variable"

    # attributes

    initValue = attr_def(Attribute)
    name_var = attr_def(StringAttr)

    # results

    result = var_result_def(Attribute)

    def __init__(
        self,
        initValue: Attribute,
        name_var: str,
        result: Sequence[Attribute],
    ):
        attributes: dict[str, Attribute] = {}
        attributes["initValue"] = initValue
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
