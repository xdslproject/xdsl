from __future__ import annotations

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    ArrayAttr,
    ContainerOf,
    DictionaryAttr,
    FlatSymbolRefAttr,
    # ParametrizedAttribute,
    FunctionType,
    IndexType,
    # InstanceType,
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
    # irdl_attr_definition,
    operand_def,
    opt_attr_def,
    opt_region_def,
    region_def,
    var_operand_def,
    var_result_def,
)

# result_def,
# ParameterDef,
from xdsl.traits import HasParent, NoTerminator, SymbolOpInterface, SymbolTable
from xdsl.utils.exceptions import VerifyException

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))


@irdl_op_definition
class HWInstance(IRDLOperation):
    """represents a hardware-style instance of a state machine,
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
        sym_name: str,
        machine: FlatSymbolRefAttr,
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
        attributes["sym_name"] = StringAttr(sym_name)
        attributes["machine"] = machine
        super().__init__(
            operands=[inputs, clock, reset],
            result_types=[outputs],
            attributes=attributes,
        )

    def verify_(self):
        found = False
        t1 = type(self.inputs)
        for d in getattr(SymbolTable.lookup_symbol(self, self.machine), "arg_names"):
            if t1 == type(d):
                found = True
                break
        if not found:
            raise VerifyException("Input is not consistent with machine inputs")
        t1 = type(self.outputs)
        found = False
        for d in getattr(SymbolTable.lookup_symbol(self, self.machine), "res_names"):
            if t1 == type(d):
                found = True
                break
        if not found:
            raise VerifyException("Output is not consistent with machine outputs")


@irdl_op_definition
class Instance(IRDLOperation):
    """represents an instance of a state machine, including an
    instance name and a symbol reference of the machine"""

    name = "fsm.instance"

    sym_name = attr_def(StringAttr)
    machine = attr_def(FlatSymbolRefAttr)

    def __init__(
        self, sym_name: str, machine: FlatSymbolRefAttr, instance: Sequence[Attribute]
    ):
        if isinstance(machine, str):
            machine = FlatSymbolRefAttr(machine)
        attributes: dict[str, Attribute] = {}
        attributes["sym_name"] = StringAttr(sym_name)
        attributes["machine"] = machine
        super().__init__(result_types=[instance], attributes=attributes)

    # verify that an insance and machine exist would make sense (to me)
    # also verifying that machines are consistent in instance and operator

    def verify_(self):
        assert isinstance(SymbolTable.lookup_symbol(self, self.machine), Machine)
        if SymbolTable.lookup_symbol(self, self.machine) is None:
            raise VerifyException("The machine does not exist")
        SymbolTable.lookup_symbol(self, self.machine)


@irdl_op_definition
class Machine(IRDLOperation):
    """represents a finite-state machine, including a machine name,
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


@irdl_op_definition
class Output(IRDLOperation):
    """represents the outputs of a machine under a specific state. The
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
        # res_type_machine = type(None)
        while HasParent(Machine):
            # if isinstance(parent, Machine):
            #     res_type_machine = type(parent.arg_attrs)
            if parent is not None:
                parent = parent.parent_op()
            else:
                raise VerifyException("Output must be in a machine")
            # if res_type_machine is not type(None):
            #     if not res_type_machine == type(self.operands):
            #         raise VerifyException(
            #             "Output type must be consistent with the machine"
            #         )


@irdl_op_definition
class Return(IRDLOperation):
    """arks the end of a region of `fsm.transition` and return
    values if the parent region is a `$guard` region"""

    name = "fsm.return"

    operand = operand_def(signlessIntegerLike)

    def __init__(
        self,
        operand: SSAValue | Operation,
    ):
        super().__init__(
            operands=[operand],
        )

    def verify_(self) -> None:
        if not self.operands:
            return

        parent = self.parent_op()
        if not isinstance(parent, Transition):
            return

        if self.parent_region() != parent.guard:
            raise VerifyException(
                "Only returns values if the parent region is a guard region"
            )


@irdl_op_definition
class State(IRDLOperation):
    """represents a state of a state machine. This op includes an
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
    """represents a transition of a state with a symbol reference
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
    """triggers a state machine instance. The inputs and outputs are
    correponding to the inputs and outputs of the referenced machine of the
    instance"""

    name = "fsm.trigger"

    # operands

    inputs = var_operand_def(AnyAttr())

    instance = operand_def()

    outputs = var_result_def(AnyAttr())

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        instance: SSAValue | Operation,
        outputs: Sequence[Attribute],
    ):
        super().__init__(
            operands=[inputs, instance],
            result_types=[outputs],
        )

    # def verify_(self):
    #     if not (
    #         type(getattr(SymbolOpInterface(self.instance), "arg_names"))
    #         == type(self.inputs)
    #     ):
    #         raise VerifyException("Input is not consistent with machine inputs")
    #     if not (
    #         type(getattr(SymbolOpInterface(self.instance), "res_names"))
    #         == type(self.outputs)
    #     ):
    #         raise VerifyException("Output is not consistent with machine outputs")
    # assert isinstance(SymbolTable.lookup_symbol(self, self.instance), Instance)
    # assert isinstance(SymbolTable.lookup_symbol(self, self.instance), InstanceType)
    # if SymbolTable.lookup_symbol(self, self.instance) is None:
    #     raise VerifyException("The instance does not exist")


@irdl_op_definition
class Update(IRDLOperation):
    """updates the `$variable` with the `$value`. The definition op of
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
    """represents an internal variable in a state machine with an
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
