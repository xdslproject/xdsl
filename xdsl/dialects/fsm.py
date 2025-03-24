"""
Implementation of the FSM dialect by CIRCT

See external [documentation](https://circt.llvm.org/docs/Dialects/FSM/RationaleFSM/).
"""

from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.arith import signlessIntegerLike
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FlatSymbolRefAttr,
    FlatSymbolRefAttrConstr,
    FunctionType,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    region_def,
    result_def,
    traits_def,
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


@irdl_attr_definition
class InstanceType(ParametrizedAttribute, TypeAttribute):
    name = "fsm.instancetype"


@irdl_op_definition
class MachineOp(IRDLOperation):
    """
    Represents a finite-state machine, including a machine name,
    the type of machine state, and the types of inputs and outputs. This op also
    includes a `$body` region that contains internal variables and states.
    """

    name = "fsm.machine"

    body = region_def()

    sym_name = attr_def(StringAttr)
    initialState = attr_def(StringAttr)
    function_type = attr_def(FunctionType)
    arg_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    arg_names = opt_attr_def(ArrayAttr[StringAttr])
    res_names = opt_attr_def(ArrayAttr[StringAttr])

    traits = traits_def(NoTerminator(), SymbolTable(), SymbolOpInterface())

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
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)

        attributes: dict[str, Attribute | None] = {
            "sym_name": StringAttr(sym_name),
            "initialState": StringAttr(initial_state),
            "function_type": function_type,
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
            "arg_names": arg_names,
            "res_names": res_names,
        }
        if not isinstance(body, Region):
            body = Region(Block())

        super().__init__(attributes=attributes, regions=[body])

    def verify_(self):
        if not SymbolTable.lookup_symbol(self, self.initialState):
            raise VerifyException("Can not find initial state")
        if (self.arg_attrs is None) ^ (self.arg_names is None):
            raise VerifyException("arg_attrs must be consistent with arg_names")
        elif self.arg_attrs is not None and self.arg_names is not None:
            if len(self.arg_attrs) != len(self.arg_names):
                raise VerifyException(
                    "The number of arg_attrs and arg_names should be the same"
                )
        if (self.res_attrs is None) ^ (self.res_names is None):
            raise VerifyException("res_attrs must be consistent with res_names")
        elif self.res_attrs is not None and self.res_names is not None:
            if len(self.res_attrs) != len(self.res_names):
                raise VerifyException(
                    "The number of res_attrs and res_names should be the same"
                )


@irdl_op_definition
class StateOp(IRDLOperation):
    """
    Represents a state of a state machine. This op includes an
    `$output` region with an `fsm.output` as terminator to define the machine
    outputs under this state. This op also includes a `transitions` region that
    contains all the transitions of this state
    """

    name = "fsm.state"

    output = region_def()

    transitions = region_def()

    sym_name = attr_def(StringAttr)

    traits = traits_def(NoTerminator(), SymbolOpInterface(), HasParent(MachineOp))

    def __init__(
        self,
        sym_name: str,
        output: Region | type[Region.DEFAULT] = Region.DEFAULT,
        transitions: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(sym_name),
        }
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
        assert isinstance(parent, MachineOp)
        if (
            parent.res_attrs is not None
            and len(parent.res_attrs) > 0
            and self.output.block.first_op is None
        ):
            raise VerifyException(
                "State must have a non-empty output region when the machine has results."
            )
        parent = parent.parent_op()


@irdl_op_definition
class OutputOp(IRDLOperation):
    """
    Represents the outputs of a machine under a specific state. The
    types of `$operands` should be consistent with the output types of the state
    machine
    """

    name = "fsm.output"

    operand = var_operand_def()

    traits = traits_def(IsTerminator(), HasParent(StateOp))

    def __init__(
        self,
        operand: Sequence[SSAValue | Operation],
    ):
        super().__init__(
            operands=[operand],
        )

    def verify_(self):
        parent = self.parent_op()
        assert isinstance(parent, StateOp)
        if parent.transitions == self.parent_region() and len(self.operands) > 0:
            raise VerifyException("Transition regions should not output any value")
        while (parent := parent.parent_op()) is not None:
            if isinstance(parent, MachineOp):
                if self.operand_types != parent.function_type.outputs.data:
                    raise VerifyException(
                        "OutputOp output type must be consistent with the machine "
                        + str(parent.sym_name)
                    )


@irdl_op_definition
class TransitionOp(IRDLOperation):
    """
    Represents a transition of a state with a symbol reference
    of the next state. This op includes an optional `$guard` region with an `fsm.return`
    as terminator that returns a Boolean value indicating the guard condition of
    this transition. This op also includes an optional `$action` region that represents
    the actions to be executed when this transition is taken
    """

    name = "fsm.transition"

    guard = region_def()

    action = region_def()

    nextState = attr_def(FlatSymbolRefAttrConstr)

    traits = traits_def(NoTerminator(), HasParent(StateOp))

    def __init__(
        self,
        nextState: FlatSymbolRefAttr,
        guard: Region | type[Region.DEFAULT] = Region.DEFAULT,
        action: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        if isinstance(nextState, str):
            nextState = SymbolRefAttr(nextState)
        attributes: dict[str, Attribute] = {
            "nextState": nextState,
        }
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
            SymbolTable.lookup_symbol(self, self.nextState), StateOp
        ):
            raise VerifyException("Can not find next state")
        if self.guard.blocks and not isinstance(self.guard.block.last_op, ReturnOp):
            raise VerifyException("Guard region must terminate with ReturnOp")
        var = self.parent_op()
        assert isinstance(var, StateOp)
        if var.transitions != self.parent_region():
            raise VerifyException("Transition must be located in a transitions region")


@irdl_op_definition
class UpdateOp(IRDLOperation):
    """
    Updates the `$variable` with the `$value`. The definition op of
    `$variable` should be an `fsm.variable`. This op should *only* appear in the
    `action` region of a transtion
    """

    name = "fsm.update"

    variable = operand_def(Attribute)

    value = operand_def(Attribute)

    traits = traits_def(HasParent(TransitionOp))

    def __init__(
        self,
        variable: SSAValue | Operation,
        value: SSAValue | Operation,
    ):
        super().__init__(
            operands=[variable, value],
        )

    def verify_(self) -> None:
        if not isinstance(self.variable.owner, VariableOp):
            raise VerifyException("Destination is not a variable operation")

        parent = self.parent_op()
        assert isinstance(parent, TransitionOp)

        found = 0
        for op in parent.action.walk():
            if isinstance(op, UpdateOp) and op.variable == self.variable:
                found += 1
        if found == 0:
            raise VerifyException(
                "Update must only be located in the action region of a transition"
            )
        elif found > 1:
            raise VerifyException(
                "Multiple updates to the same variable within a single action region is disallowed"
            )


@irdl_op_definition
class VariableOp(IRDLOperation):
    """
    Represents an internal variable in a state machine with an
    initialization value
    """

    name = "fsm.variable"

    initValue = attr_def(Attribute)
    name_var = opt_attr_def(StringAttr)

    result = var_result_def(Attribute)

    def __init__(
        self,
        initValue: Attribute,
        name_var: str | None,
        result: Sequence[Attribute],
    ):
        attributes: dict[str, Attribute] = {
            "initValue": initValue,
        }
        if name_var is not None:
            attributes["name_var"] = StringAttr(name_var)
        super().__init__(
            result_types=[result],
            attributes=attributes,
        )


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """
    Marks the end of a region of `fsm.transition` and return
    values if the parent region is a `$guard` region
    """

    name = "fsm.return"

    operand = opt_operand_def(signlessIntegerLike)

    traits = traits_def(IsTerminator(), HasParent(TransitionOp))

    def __init__(
        self,
        operand: SSAValue | Operation,
    ):
        super().__init__(
            operands=[operand],
        )


@irdl_op_definition
class InstanceOp(IRDLOperation):
    """
    Represents an instance of a state machine, including an
    instance name and a symbol reference of the machine
    """

    name = "fsm.instance"

    sym_name = attr_def(StringAttr)

    machine = attr_def(FlatSymbolRefAttrConstr)

    res = result_def(InstanceType)

    def __init__(
        self, sym_name: str, machine: FlatSymbolRefAttr, instance: InstanceType
    ):
        if isinstance(machine, str):
            machine = SymbolRefAttr(machine)
        attributes: dict[str, Attribute] = {
            "sym_name": StringAttr(sym_name),
            "machine": machine,
        }
        super().__init__(result_types=[instance], attributes=attributes)

    def verify_(self):
        if not isinstance(SymbolTable.lookup_symbol(self, self.machine), MachineOp):
            raise VerifyException("Machine definition does not exist")

    def getMachine(self) -> MachineOp | None:
        m = SymbolTable.lookup_symbol(self, self.machine)
        if isinstance(m, MachineOp):
            return m
        else:
            return None


@irdl_op_definition
class TriggerOp(IRDLOperation):
    """
    Triggers a state machine instance. The inputs and outputs are
    correponding to the inputs and outputs of the referenced machine of the
    instance
    """

    name = "fsm.trigger"

    inputs = var_operand_def()

    instance = operand_def(InstanceType)

    outputs = var_result_def()

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
        if not isinstance(self.instance.owner, InstanceOp):
            raise VerifyException("The instance operand must be Instance")

        m = self.instance.owner.getMachine()

        if m is None:
            raise VerifyException("Machine definition does not exist.")

        if self.inputs.types != tuple(result for result in m.function_type.inputs):
            raise VerifyException(
                "TriggerOp input types must be consistent with the machine "
                + str(m.sym_name)
            )

        if self.outputs.types != tuple(result for result in m.function_type.outputs):
            raise VerifyException(
                "TriggerOp output types must be consistent with the machine "
                + str(m.sym_name)
            )


@irdl_op_definition
class HWInstanceOp(IRDLOperation):
    """
    Represents a hardware-style instance of a state machine,
    including an instance name and a symbol reference of the machine. The inputs
    and outputs are correponding to the inputs and outputs of the referenced
    machine.
    """

    name = "fsm.hw_instance"

    sym_name = attr_def(StringAttr)
    machine = attr_def(FlatSymbolRefAttrConstr)
    inputs = var_operand_def()
    clock = operand_def(signlessIntegerLike)
    reset = operand_def(signlessIntegerLike)

    outputs = var_result_def()

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
            machine = SymbolRefAttr(machine)
        clock = SSAValue.get(clock)
        reset = SSAValue.get(reset)
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        attributes: dict[str, Attribute] = {
            "sym_name": sym_name,
            "machine": machine,
        }
        super().__init__(
            operands=[inputs, clock, reset],
            result_types=[outputs],
            attributes=attributes,
        )

    def verify_(self):
        m = SymbolTable.lookup_symbol(self, self.machine)
        if isinstance(m, MachineOp):
            if self.inputs.types != tuple(result for result in m.function_type.inputs):
                raise VerifyException(
                    "HWInstanceOp "
                    + str(self.sym_name)
                    + " input type must be consistent with the machine "
                    + str(m.sym_name)
                )
            if self.outputs.types != tuple(
                result for result in m.function_type.outputs
            ):
                raise VerifyException(
                    "HWInstanceOp "
                    + str(self.sym_name)
                    + " output type must be consistent with the machine "
                    + str(m.sym_name)
                )
        else:
            raise VerifyException("Machine definition does not exist")


FSM = Dialect(
    "fsm",
    [
        MachineOp,
        OutputOp,
        StateOp,
        TransitionOp,
        UpdateOp,
        VariableOp,
        ReturnOp,
        TriggerOp,
        InstanceOp,
        HWInstanceOp,
    ],
    [InstanceType],
)
