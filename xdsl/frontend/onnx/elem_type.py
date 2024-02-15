from xdsl.dialects.builtin import f32, f64
from xdsl.ir import Attribute

ELEM_TYPE = {
    1: f32,
    11: f64,
}


def get_elem_type(code: int) -> Attribute:
    if code in ELEM_TYPE:
        return ELEM_TYPE[code]
    else:
        raise ValueError(f"Unknown elem_type: {code}")
