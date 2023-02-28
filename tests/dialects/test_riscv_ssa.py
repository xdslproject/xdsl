from xdsl.dialects import riscv_ssa, builtin


def test_Riscv1Rd1Rs1ImmOperation():
    val = riscv_ssa.LIOp.get(0)
    lb_op = riscv_ssa.LBOp.get(val, 100)

    assert isinstance(lb_op.immediate, builtin.IntAttr)
    assert lb_op.immediate.data == 100
    assert lb_op.rs1 == val.rd

    lb_op_str = riscv_ssa.LBOp.get(val, "label", "Some Comment")
    assert isinstance(lb_op_str, riscv_ssa.Riscv1Rd1Rs1ImmOperation)

    assert isinstance(lb_op_str.immediate, riscv_ssa.LabelAttr)
    assert lb_op_str.immediate.string_value() == "label"
    assert isinstance(lb_op_str.comment, builtin.StringAttr)
    assert lb_op_str.comment.data == "Some Comment"


def test_Riscv2Rs1ImmOperation():
    val = riscv_ssa.LIOp.get(0)

    op = riscv_ssa.SWOp.get(val, val, 100)

    assert isinstance(op.immediate, builtin.IntAttr)
    assert isinstance(op, riscv_ssa.Riscv2Rs1ImmOperation)
    assert op.immediate.data == 100
    assert op.rs1 == val.rd
    assert op.rs2 == val.rd

    op = riscv_ssa.SWOp.get(val, val, "label")

    assert isinstance(op.immediate, riscv_ssa.LabelAttr)
    assert op.immediate.label.data == "label"
    assert op.rs1 == val.rd
    assert op.rs2 == val.rd


def test_Riscv2Rs1OffOperation():
    val = riscv_ssa.LIOp.get(0)
    op = riscv_ssa.BEQOp.get(val, val, 100)

    assert isinstance(op.offset, builtin.IntAttr)
    assert isinstance(op, riscv_ssa.Riscv2Rs1OffOperation)
    assert op.offset.data == 100
    assert op.rs1 == val.rd
    assert op.rs2 == val.rd

    op = riscv_ssa.BEQOp.get(val, val, "label", 'comment')

    assert isinstance(op.offset, riscv_ssa.LabelAttr)
    assert op.offset.string_value() == 'label'
    assert op.rs1 == val.rd
    assert op.rs2 == val.rd


def test_Riscv1Rd2RsOperation():
    val = riscv_ssa.LIOp.get(0)

    op = riscv_ssa.SLLOp.get(val, val, 'comment')
    assert isinstance(op, riscv_ssa.Riscv1Rd2RsOperation)
    assert op.rs1 == val.rd
    assert op.rs2 == val.rd
    assert op.comment is not None
    assert op.comment.data == 'comment'


def test_Riscv1OffOperation():
    op = riscv_ssa.JOp.get(100)
    assert isinstance(op, riscv_ssa.Riscv1OffOperation)
    assert isinstance(op.offset, builtin.IntAttr)
    assert op.offset.data == 100

    op = riscv_ssa.JOp.get('label')
    assert isinstance(op.offset, riscv_ssa.LabelAttr)
    assert op.offset.string_value() == 'label'


def test_Riscv1Rd1ImmOperation():
    op = riscv_ssa.LUIOp.get(100)
    assert isinstance(op, riscv_ssa.Riscv1Rd1ImmOperation)
    assert isinstance(op.immediate, builtin.IntAttr)
    assert op.immediate.data == 100

    op = riscv_ssa.LUIOp.get('label')
    assert isinstance(op.immediate, riscv_ssa.LabelAttr)
    assert op.immediate.string_value() == 'label'


def test_ECALLOp():
    # TODO
    pass


def test_CallOp():
    # TODO
    pass


def test_LabelOp():
    # TODO
    pass


def test_DirectiveOp():
    # TODO
    pass


def test_FuncOp():
    # TODO
    pass


def test_ReturnOp():
    # TODO
    pass


def test_SectionOp():
    # TODO
    pass
