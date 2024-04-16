from xdsl.irdl import IRDLOperation, irdl_op_definition

# fmt: off

# ANCHOR: test_op
@irdl_op_definition
class SomeTestOp(IRDLOperation):
    name = "some.test"
# ANCHOR_END: test_op

raise ValueError("oh no this is not working")
