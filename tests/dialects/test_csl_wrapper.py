from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.dialects.csl import csl_wrapper


def test_get_layout_arg():
    m_op = csl_wrapper.ModuleOp(
        7, 8, params={"one": IntegerAttr(9, 16), "two": IntegerAttr(10, 16)}
    )
    assert m_op.get_layout_param("x") == m_op.layout_module.block.args[0]
    assert m_op.get_layout_param("y") == m_op.layout_module.block.args[1]
    assert m_op.get_layout_param("width") == m_op.layout_module.block.args[2]
    assert m_op.get_layout_param("height") == m_op.layout_module.block.args[3]
    assert m_op.get_layout_param("one") == m_op.layout_module.block.args[4]
    assert m_op.get_layout_param("two") == m_op.layout_module.block.args[5]
    assert len(m_op.layout_module.block.args) == 6


def test_get_program_arg():
    m_op = csl_wrapper.ModuleOp(
        7, 8, params={"one": IntegerAttr(9, 16), "two": IntegerAttr(10, 16)}
    )
    assert m_op.get_program_param("width") == m_op.program_module.block.args[0]
    assert m_op.get_program_param("height") == m_op.program_module.block.args[1]
    assert m_op.get_program_param("one") == m_op.program_module.block.args[2]
    assert m_op.get_program_param("two") == m_op.program_module.block.args[3]
    assert len(m_op.program_module.block.args) == 4


def test_update_program_args():
    m_op = csl_wrapper.ModuleOp(
        7, 8, params={"one": IntegerAttr(9, 16), "two": IntegerAttr(10, 16)}
    )
    assert len(m_op.program_module.block.args) == 4
    with ImplicitBuilder(m_op.layout_module.block):
        zero_const = arith.Constant(IntegerAttr(0, 16))
        seven_const = arith.Constant(IntegerAttr(7, 32))
        csl_wrapper.YieldOp.from_field_name_mapping(
            {
                "zero_param": zero_const,
                "seven_param": seven_const,
            }
        )

    m_op.update_program_block_args()

    assert len(m_op.program_module.block.args) == 6
    assert m_op.get_program_param("width") == m_op.program_module.block.args[0]
    assert m_op.get_program_param("height") == m_op.program_module.block.args[1]
    assert m_op.get_program_param("one") == m_op.program_module.block.args[2]
    assert m_op.get_program_param("two") == m_op.program_module.block.args[3]
    assert m_op.get_program_param("zero_param") == m_op.program_module.block.args[4]
    assert m_op.get_program_param("seven_param") == m_op.program_module.block.args[5]
    assert m_op.program_module.block.args[4].type == IntegerType(16)
    assert m_op.program_module.block.args[5].type == IntegerType(32)


def test_exported_symbols():
    m_op = csl_wrapper.ModuleOp(
        7, 8, params={"one": IntegerAttr(9, 16), "two": IntegerAttr(10, 16)}
    )
    assert len(m_op.program_module.block.args) == 4
    with ImplicitBuilder(m_op.layout_module.block):
        zero_const = arith.Constant(IntegerAttr(0, 16))
        seven_const = arith.Constant(IntegerAttr(7, 32))
        csl_wrapper.YieldOp.from_field_name_mapping(
            {
                "zero_param": zero_const,
                "seven_param": seven_const,
            }
        )

    m_op.update_program_block_args(
        exported_symbols=[("a_export_arg", IntegerType(17)), (None, IntegerType(18))]
    )

    assert len(m_op.program_module.block.args) == 8
    assert m_op.get_program_param("width") == m_op.program_module.block.args[0]
    assert m_op.get_program_param("height") == m_op.program_module.block.args[1]
    assert m_op.get_program_param("one") == m_op.program_module.block.args[2]
    assert m_op.get_program_param("two") == m_op.program_module.block.args[3]
    assert m_op.get_program_param("zero_param") == m_op.program_module.block.args[4]
    assert m_op.get_program_param("seven_param") == m_op.program_module.block.args[5]
    assert m_op.program_module.block.args[4].type == IntegerType(16)
    assert m_op.program_module.block.args[5].type == IntegerType(32)

    # test exported symbols
    assert len(m_op.exported_symbols) == 2
    assert m_op.exported_symbols[0] == m_op.program_module.block.args[6]
    assert m_op.exported_symbols[1] == m_op.program_module.block.args[7]
    assert m_op.program_module.block.args[6].type == IntegerType(17)
    assert m_op.program_module.block.args[7].type == IntegerType(18)
