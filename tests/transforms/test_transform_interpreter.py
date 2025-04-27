import pytest

from xdsl.dialects import builtin
from xdsl.transforms.transform_interpreter import TransformInterpreterPass
from xdsl.utils.exceptions import PassFailedException


def test_entry_point_not_found():
    root = builtin.ModuleOp(ops=[])
    entry_point = "symbol-name"
    msg = f"could not find a nested named sequence with name: {entry_point}"
    with pytest.raises(PassFailedException, match=msg):
        TransformInterpreterPass.find_transform_entry_point(root, "symbol-name")
