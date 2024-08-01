import pytest

from xdsl.passes import ModulePass
from xdsl.transforms import (
    convert_riscv_scf_for_to_frep,
    memref_stream_interleave,
    memref_stream_unnest_out_parameters,
    memref_streamify,
)
from xdsl.transforms.test_lower_linalg_to_snitch import get_excluded_passes


@pytest.mark.parametrize(
    "optimization_level,expected",
    [
        (
            0,
            (
                memref_stream_interleave.MemrefStreamInterleavePass(),
                convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
                memref_stream_unnest_out_parameters.MemrefStreamUnnestOutParametersPass(),
                memref_streamify.MemrefStreamifyPass(),
            ),
        ),
        (
            1,
            (
                memref_stream_interleave.MemrefStreamInterleavePass(),
                convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
                memref_stream_unnest_out_parameters.MemrefStreamUnnestOutParametersPass(),
            ),
        ),
        (
            2,
            (
                memref_stream_interleave.MemrefStreamInterleavePass(),
                convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
            ),
        ),
        (
            3,
            (memref_stream_interleave.MemrefStreamInterleavePass(),),
        ),
        (4, ()),
    ],
)
def test_get_excluded_passes(optimization_level: int, expected: tuple[ModulePass, ...]):
    assert get_excluded_passes(optimization_level) == expected
