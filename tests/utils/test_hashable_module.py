from collections.abc import Iterable

from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.test import TestOp
from xdsl.utils.hashable_module import HashableModule


def _gen_module(labels: Iterable[str]):
    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        for label in labels:
            TestOp(attributes={"label": StringAttr(label)})
    return module


def test_hashable_module():
    assert HashableModule(ModuleOp([])) == HashableModule(ModuleOp([]))
    assert hash(HashableModule(ModuleOp([]))) == hash(HashableModule(ModuleOp([])))

    a0 = _gen_module("a")
    a1 = _gen_module("a")

    assert HashableModule(a0) == HashableModule(a1)
    assert hash(HashableModule(a0)) == hash(HashableModule(a1))

    abra0 = _gen_module("abra")
    abra1 = _gen_module("abra")

    assert HashableModule(abra0) == HashableModule(abra1)
    assert hash(HashableModule(abra0)) == hash(HashableModule(abra1))

    assert HashableModule(a0) != HashableModule(abra1)
