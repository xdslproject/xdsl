from xdsl.dialects.x86 import GeneralRegisterType
from xdsl.dialects.x86.ops import (
    DI_Operation,
    DM_Operation,
    DMI_Operation,
    DS_Operation,
    DSI_Operation,
    DSSI_Operation,
    M_Operation,
    MI_Operation,
    MS_Operation,
    R_Operation,
    RI_Operation,
    RM_Operation,
    RS_Operation,
    RSS_Operation,
)
from xdsl.irdl import irdl_op_definition
from xdsl.utils import test_value

reg_type = GeneralRegisterType.infinite_register


@irdl_op_definition
class TestRSOperation(RS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from RS_Operation for testing register constraints."""

    name = "test.rs_operation"


def test_rs_operation_register_constraints():
    rs_op = TestRSOperation(
        test_value.create_ssa_value(reg_type(0)),
        test_value.create_ssa_value(reg_type(1)),
    )

    rs_c = rs_op.get_register_constraints()

    assert rs_c.ins == (rs_op.source,)
    assert rs_c.outs == ()
    assert rs_c.inouts == ((rs_op.register_in, rs_op.register_out),)


@irdl_op_definition
class TestROperation(R_Operation[GeneralRegisterType]):
    """Test operation that inherits from R_Operation for testing register constraints."""

    name = "test.r_operation"


def test_r_operation_register_constraints():
    r_op = TestROperation(
        test_value.create_ssa_value(reg_type(0)),
    )

    r_c = r_op.get_register_constraints()

    assert r_c.ins == ()
    assert r_c.outs == ()
    assert r_c.inouts == ((r_op.register_in, r_op.register_out),)


@irdl_op_definition
class TestRMOperation(RM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from RM_Operation for testing register constraints."""

    name = "test.rm_operation"


def test_rm_operation_register_constraints():
    # Create an instance of our test RM_Operation
    rm_op = TestRMOperation(
        test_value.create_ssa_value(reg_type(0)),
        test_value.create_ssa_value(reg_type(1)),
        memory_offset=42,
        register_out=reg_type(2),
    )

    rm_c = rm_op.get_register_constraints()

    assert rm_c.ins == (rm_op.memory,)
    assert rm_c.outs == ()
    assert rm_c.inouts == ((rm_op.register_in, rm_op.register_out),)


@irdl_op_definition
class TestDMOperation(DM_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from DM_Operation for testing register constraints."""

    name = "test.dm_operation"


def test_dm_operation_register_constraints():
    # Create an instance of our test DM_Operation
    dm_op = TestDMOperation(
        test_value.create_ssa_value(reg_type(0)),
        memory_offset=42,
        destination=reg_type(1),
    )

    dm_c = dm_op.get_register_constraints()

    assert tuple(dm_c.ins) == (dm_op.memory,)
    assert dm_c.outs == (dm_op.destination,)
    assert dm_c.inouts == ()


@irdl_op_definition
class TestRIOperation(RI_Operation[GeneralRegisterType]):
    """Test operation that inherits from RI_Operation for testing register constraints."""

    name = "test.ri_operation"


def test_ri_operation_register_constraints():
    # Create an instance of our test RI_Operation
    ri_op = TestRIOperation(
        test_value.create_ssa_value(reg_type(0)),
        immediate=42,
        register_out=reg_type(1),
    )

    ri_c = ri_op.get_register_constraints()

    assert ri_c.ins == ()
    assert ri_c.outs == ()
    assert ri_c.inouts == ((ri_op.register_in, ri_op.register_out),)


@irdl_op_definition
class TestMSOperation(MS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from MS_Operation for testing register constraints."""

    name = "test.ms_operation"


def test_ms_operation_register_constraints():
    # Create an instance of our test MS_Operation
    ms_op = TestMSOperation(
        test_value.create_ssa_value(reg_type(0)),
        test_value.create_ssa_value(reg_type(1)),
        memory_offset=42,
    )

    ms_c = ms_op.get_register_constraints()

    assert tuple(ms_c.ins) == (
        ms_op.memory,
        ms_op.source,
    )
    assert ms_c.outs == ()
    assert ms_c.inouts == ()


@irdl_op_definition
class TestMIOperation(MI_Operation[GeneralRegisterType]):
    """Test operation that inherits from MI_Operation for testing register constraints."""

    name = "test.mi_operation"


def test_mi_operation_register_constraints():
    # Create an instance of our test MI_Operation
    mi_op = TestMIOperation(
        test_value.create_ssa_value(reg_type(0)),
        immediate=42,
        memory_offset=10,
    )

    mi_c = mi_op.get_register_constraints()

    assert tuple(mi_c.ins) == (mi_op.memory,)
    assert mi_c.outs == ()
    assert mi_c.inouts == ()


@irdl_op_definition
class TestDSIOperation(DSI_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from DSI_Operation for testing register constraints."""

    name = "test.dsi_operation"


def test_dsi_operation_register_constraints():
    # Create an instance of our test DSI_Operation
    dsi_op = TestDSIOperation(
        test_value.create_ssa_value(reg_type(0)),
        immediate=42,
        destination=reg_type(1),
    )

    dsi_c = dsi_op.get_register_constraints()

    assert tuple(dsi_c.ins) == (dsi_op.source,)
    assert dsi_c.outs == (dsi_op.destination,)
    assert dsi_c.inouts == ()


@irdl_op_definition
class TestDMIOperation(DMI_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from DMI_Operation for testing register constraints."""

    name = "test.dmi_operation"


def test_dmi_operation_register_constraints():
    # Create an instance of our test DMI_Operation
    dmi_op = TestDMIOperation(
        test_value.create_ssa_value(reg_type(0)),
        memory_offset=10,
        immediate=42,
        destination=reg_type(1),
    )

    dmi_c = dmi_op.get_register_constraints()

    assert tuple(dmi_c.ins) == (dmi_op.memory,)
    assert dmi_c.outs == (dmi_op.destination,)
    assert dmi_c.inouts == ()


@irdl_op_definition
class TestMOperation(M_Operation[GeneralRegisterType]):
    """Test operation that inherits from M_Operation for testing register constraints."""

    name = "test.m_operation"


def test_m_operation_register_constraints():
    # Create an instance of our test M_Operation
    m_op = TestMOperation(
        test_value.create_ssa_value(reg_type(0)),
        memory_offset=10,
    )

    m_c = m_op.get_register_constraints()

    assert tuple(m_c.ins) == (m_op.memory,)
    assert m_c.outs == ()
    assert m_c.inouts == ()


@irdl_op_definition
class TestRSSOperation(
    RSS_Operation[GeneralRegisterType, GeneralRegisterType, GeneralRegisterType]
):
    """Test operation that inherits from RSS_Operation for testing register constraints."""

    name = "test.rss_operation"


def test_rss_operation_register_constraints():
    # Create an instance of our test RSS_Operation
    rss_op = TestRSSOperation(
        test_value.create_ssa_value(reg_type(0)),
        test_value.create_ssa_value(reg_type(1)),
        test_value.create_ssa_value(reg_type(2)),
    )

    rss_c = rss_op.get_register_constraints()

    assert tuple(rss_c.ins) == (rss_op.source1, rss_op.source2)
    assert rss_c.outs == ()
    assert rss_c.inouts == ((rss_op.register_in, rss_op.register_out),)


@irdl_op_definition
class TestDSSIOperation(
    DSSI_Operation[GeneralRegisterType, GeneralRegisterType, GeneralRegisterType]
):
    """Test operation that inherits from DSSI_Operation for testing register constraints."""

    name = "test.dssi_operation"


def test_irs_operation_register_constraints():
    # Create an instance of our test RSS_Operation
    dssi_op = TestDSSIOperation(
        test_value.create_ssa_value(reg_type(1)),
        test_value.create_ssa_value(reg_type(2)),
        5,
        destination=reg_type(0),
    )

    dssi_c = dssi_op.get_register_constraints()

    assert tuple(dssi_c.ins) == (dssi_op.source0, dssi_op.source1)
    assert dssi_c.outs == (dssi_op.destination,)
    assert dssi_c.inouts == ()


@irdl_op_definition
class TestDSOperation(DS_Operation[GeneralRegisterType, GeneralRegisterType]):
    """Test operation that inherits from DS_Operation for testing register constraints."""

    name = "test.ds_operation"


def test_ds_operation_register_constraints():
    # Create an instance of our test DS_Operation
    ds_op = TestDSOperation(
        test_value.create_ssa_value(reg_type(0)),
        destination=reg_type(1),
    )

    ds_c = ds_op.get_register_constraints()

    assert tuple(ds_c.ins) == (ds_op.source,)
    assert ds_c.outs == (ds_op.destination,)
    assert ds_c.inouts == ()


@irdl_op_definition
class TestDIOperation(DI_Operation[GeneralRegisterType]):
    """Test operation that inherits from DI_Operation for testing register constraints."""

    name = "test.di_operation"


def test_di_operation_register_constraints():
    # Create an instance of our test DI_Operation
    di_op = TestDIOperation(
        10,
        destination=reg_type(0),
    )

    di_c = di_op.get_register_constraints()

    assert tuple(di_c.ins) == ()
    assert di_c.outs == (di_op.destination,)
    assert di_c.inouts == ()
