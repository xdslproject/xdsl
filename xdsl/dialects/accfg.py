from __future__ import annotations

from collections.abc import Iterable, Sequence

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    StringAttr,
    SymbolRefAttr,
    i32,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    VerifyException,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    prop_def,
    result_def,
    var_operand_def,
)
from xdsl.traits import SymbolOpInterface


@irdl_attr_definition
class TokenType(ParametrizedAttribute, TypeAttribute):
    """
    Async token type for managing synchronization between asynchronously
    launched accelerators. You can use them to create def-use chains
    between accfg.launch and accfg.await ops.
    """

    name = "accfg.token"

    accelerator: ParameterDef[StringAttr]

    def __init__(self, accelerator: str | StringAttr):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)
        return super().__init__([accelerator])


@irdl_attr_definition
class StateType(ParametrizedAttribute, TypeAttribute):
    """
    State type to manage accelerator configuration state.
    States are chained through the def-use chain to allow
    deduplication of setup calls.
    """

    name = "accfg.state"

    accelerator: ParameterDef[StringAttr]

    def __init__(self, accelerator: str | StringAttr):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)
        return super().__init__([accelerator])


class AcceleratorSymbolOpTrait(SymbolOpInterface):
    """
    Symbol op trait to define multiple accelerator names.
    """

    def get_sym_attr_name(self, op: Operation) -> StringAttr | None:
        assert isinstance(op, AcceleratorOp)
        return StringAttr(op.name_prop.string_value())


@irdl_op_definition
class AcceleratorOp(IRDLOperation):
    """
    Declares an accelerator that can be configured, launched, etc.
    `fields` is a dictionary mapping accelerator configuration names to
    registers addresses.
    """

    name = "accfg.accelerator"

    traits = frozenset([AcceleratorSymbolOpTrait()])

    name_prop = prop_def(SymbolRefAttr, prop_name="name")

    fields = prop_def(DictionaryAttr)

    launch_addr = prop_def(AnyIntegerAttr)

    # TODO: the barrier field will likely be changed in the future

    # The exact translation of accfg.await is not final,
    # and as such the information required for this on the accfg.accelerator
    # op will probably change again in the future.
    # We're looking into ways of generalizing this aspect currently,
    # but this is a thing that actually works for snitch now.

    barrier = prop_def(AnyIntegerAttr)

    def __init__(
        self,
        name: str | StringAttr | SymbolRefAttr,
        fields: dict[str, int] | DictionaryAttr,
        launch: int | AnyIntegerAttr,
        barrier: int | AnyIntegerAttr,
    ):
        if not isinstance(fields, DictionaryAttr):
            fields = DictionaryAttr(
                {name: IntegerAttr(val, i32) for name, val in fields.items()}
            )

        super().__init__(
            properties={
                "name": (
                    SymbolRefAttr(name) if not isinstance(name, SymbolRefAttr) else name
                ),
                "fields": fields,
                "launch_addr": (
                    IntegerAttr(launch, i32)
                    if not isinstance(launch, IntegerAttr)
                    else launch
                ),
                "barrier": (
                    IntegerAttr(barrier, i32)
                    if not isinstance(barrier, IntegerAttr)
                    else barrier
                ),
            }
        )

    def verify_(self) -> None:
        for _, val in self.fields.data.items():
            if not isinstance(val, IntegerAttr):
                raise VerifyException("fields must only contain IntegerAttr!")

    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields.data.keys())

    def field_items(self) -> Iterable[tuple[str, AnyIntegerAttr]]:
        for name, val in self.fields.data.items():
            assert isinstance(val, IntegerAttr)
            yield name, val


@irdl_op_definition
class LaunchOp(IRDLOperation):
    """
    Launch an accelerator.
    We assume that all values in configuration registers are consumed
    by the accelerator at this point. This means that
    the configuration registers to the accelerator can be reconfigured
    safely after a launch op without interfering with the Accelerator.
    """

    name = "accfg.launch"

    state = operand_def(StateType)

    accelerator = prop_def(StringAttr)

    token = result_def()

    def __init__(self, state: SSAValue | Operation):
        state_val: SSAValue = SSAValue.get(state)
        if not isinstance(state_val.type, StateType):
            raise ValueError("`state` SSA Value must be of type `accfg.state`!")
        super().__init__(
            operands=[state],
            properties={"accelerator": state_val.type.accelerator},
            result_types=[TokenType(state_val.type.accelerator)],
        )

    def verify_(self) -> None:
        # that the state and my accelerator match
        assert isinstance(self.state.type, StateType)
        if self.state.type.accelerator != self.accelerator:
            raise VerifyException(
                "The state's accelerator does not match the launch accelerator!"
            )
        # that the token and my accelerator match
        assert isinstance(self.token.type, TokenType)
        if self.token.type.accelerator != self.accelerator:
            raise VerifyException(
                "The token's accelerator does not match the launch accelerator!"
            )

        # that the token is used
        if len(self.token.uses) != 1 or not isinstance(
            next(iter(self.token.uses)).operation, AwaitOp
        ):
            raise VerifyException("Launch token must be used by exactly one await op")
        # TODO: allow use in control flow


@irdl_op_definition
class AwaitOp(IRDLOperation):
    """
    Wait until the launched operation finishes.
    Awaits the token emitted by an accfg.launch operation.
    """

    name = "accfg.await"

    token = operand_def(TokenType)

    def __init__(self, token: SSAValue | Operation):
        super().__init__(operands=[token])


@irdl_op_definition
class SetupOp(IRDLOperation):
    """
    accfg.setup writes values to a specific accelerators configuration and returns
    a value representing the currently known state of that accelerator's config.

    If accfg.setup is called without any parameters, the resulting state is the
    "empty" state, that represents a state without known values.
    """

    name = "accfg.setup"

    values = var_operand_def(Attribute)  # TODO: make more precise?
    """
    The actual values used to set up the CSRs
    """

    in_state = opt_operand_def(StateType)
    """
    The state produced by a previous accfg.setup
    """

    out_state = result_def(StateType)
    """
    The CSR state after the setup op modified it.
    """

    param_names = prop_def(ArrayAttr[StringAttr])
    """
    Maps the SSA values in `values` to accelerator parameter names
    """

    accelerator = prop_def(StringAttr)
    """
    Name of the accelerator this setup is for
    """

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        vals: Sequence[SSAValue | Operation],
        param_names: Iterable[str] | Iterable[StringAttr],
        accelerator: str | StringAttr,
        in_state: SSAValue | Operation | None = None,
    ):
        if not isinstance(accelerator, StringAttr):
            accelerator = StringAttr(accelerator)

        param_names_tuple: tuple[StringAttr, ...] = tuple(
            StringAttr(name) if isinstance(name, str) else name for name in param_names
        )

        super().__init__(
            operands=[vals, in_state],
            properties={
                "param_names": ArrayAttr(param_names_tuple),
                "accelerator": accelerator,
            },
            result_types=[StateType(accelerator)],
        )

    def iter_params(self) -> Iterable[tuple[str, SSAValue]]:
        return zip((p.data for p in self.param_names), self.values, strict=True)

    def verify_(self) -> None:
        # that accelerator on input matches output
        if self.in_state is not None:
            if self.in_state.type != self.out_state.type:
                raise VerifyException("Input and output state accelerators must match")
        assert isinstance(self.out_state.type, StateType)
        if self.accelerator != self.out_state.type.accelerator:
            raise VerifyException(
                "Output state accelerator and accelerator the "
                "operations property must match"
            )

        # that len(values) == len(param_names)
        if len(self.values) != len(self.param_names):
            raise ValueError(
                "Must have received same number of values as parameter names"
            )


ACCFG = Dialect(
    "accfg",
    [
        AcceleratorOp,
        SetupOp,
        LaunchOp,
        AwaitOp,
    ],
    [
        StateType,
        TokenType,
    ],
)
