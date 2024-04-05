"""
These inference helpers are used to figure out which values are
present at which points in the IR.

Example inference results:

```
accfg.setup(A = %1, B = %2)  // previous state = {}
// ...
accfg.setup(A = %3, C = %1)  // previous state = {A = %1, B = %2}
```

These inference passes walk the IR backwards.
"""

from xdsl.dialects import accfg, scf
from xdsl.ir import Block, SSAValue

State = dict[str, SSAValue]


def infer_state_of(state_var: SSAValue) -> State:
    """
    Entrance function of the inference pass.

    This walks up the def-use chain to compute all values
    that are guaranteed to be set in this state.
    """
    owner = state_var.owner
    match owner:
        case accfg.SetupOp(in_state=None) as setup_op:
            return {name: val for name, val in setup_op.iter_params()}
        case accfg.SetupOp(in_state=st) as setup_op if st is not None:
            in_state = infer_state_of(st)
            in_state.update(dict(setup_op.iter_params()))
            return in_state
        case scf.If() as if_op:
            return state_union(*infer_states_for_if(if_op, state_var))
        case Block():
            # TODO: maybe implement something here?
            return {}
        case _:
            raise ValueError(f"Cannot infer state for op {owner.name}")


def infer_states_for_if(op: scf.If, state: SSAValue) -> tuple[State, State]:
    """
    Walk both sides of the if/else block and return the computed
    states for the given state SSA value (`state`)
    """
    assert state in op.results, "Expected state to be one of the scf.if results!"
    idx = op.results.index(state)

    states = []
    for region in op.regions:
        # we know the last op must be the yield
        yield_op = region.block.last_op
        assert isinstance(yield_op, scf.Yield)
        # we know the yield op has the same number of operands as the
        # scf.if has results, so [idx] must be defined
        states.append(infer_state_of(yield_op.operands[idx]))
    assert len(states) == 2
    return tuple(states)


def state_union(a: State, b: State) -> State:
    return {k: a[k] for k in a if a[k] == b[k]}
