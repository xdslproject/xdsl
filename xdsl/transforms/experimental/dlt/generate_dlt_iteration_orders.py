from xdsl.dialects.experimental import dlt


def _make_nested_order(order: dlt.IterationOrder) -> dlt.IterationOrder:
    if isinstance(order, dlt.BodyIterationOrderAttr):
        return order
    elif isinstance(order, dlt.NestedIterationOrderAttr):
        child = _make_nested_order(order.child)
        return dlt.NestedIterationOrderAttr(order.extent_index, child)
    elif isinstance(order, dlt.NonZeroIterationOrderAttr):
        child = _make_nested_order(order.child)
        return dlt.NonZeroIterationOrderAttr(order.extent_index, order.tensor_index, child)
    elif isinstance(order, dlt.AbstractIterationOrderAttr):
        child = _make_nested_order(order.child)
        for extent_idx in order.extent_indices:
            child = dlt.NestedIterationOrderAttr(extent_idx, child)
        return child