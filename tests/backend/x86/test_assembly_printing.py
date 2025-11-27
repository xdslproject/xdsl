from xdsl.dialects.builtin import UnitAttr
from xdsl.dialects.x86.assembly import masked_source_str
from xdsl.dialects.x86.registers import AVX512MaskRegisterType, AVX512RegisterType
from xdsl.utils.test_value import create_ssa_value


def test_masked_source_str():
    src_reg_val = create_ssa_value(AVX512RegisterType.from_name("zmm0"))
    k_val = create_ssa_value(AVX512MaskRegisterType.from_name("k1"))

    assert masked_source_str(src_reg_val, k_val, None) == "zmm0 {k1}"
    assert masked_source_str(src_reg_val, k_val, UnitAttr()) == "zmm0 {k1}{z}"
