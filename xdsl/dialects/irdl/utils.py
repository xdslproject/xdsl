from xdsl.dialects.irdl.irdl import OperationOp


def class_name_from_op(op: OperationOp) -> str:
    return (
        "".join(
            y[:1].upper() + y[1:]
            for x in op.sym_name.data.split(".")
            for y in x.split("_")
        )
        + "Op"
    )
