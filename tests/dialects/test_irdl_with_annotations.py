from __future__ import annotations

# do not remove 'from __future__ import annotations'
from typing import ClassVar

import pytest

from xdsl.irdl import IRDLOperation, irdl_op_definition
from xdsl.utils.exceptions import PyRDLOpDefinitionError


class MyOpWithClassVar(IRDLOperation):
    name = "test.has_class_var"

    VAR: ClassVar[str] = "hello_world"


class MyOpWithClassVarInvalid(IRDLOperation):
    name = "test.has_class_var"

    var: ClassVar[str] = "hello_world"


def test_class_var_on_op():
    irdl_op_definition(MyOpWithClassVar)


def test_class_var_on_op_invalid():
    with pytest.raises(
        PyRDLOpDefinitionError, match='Invalid ClassVar name "var", must be uppercase'
    ):
        irdl_op_definition(MyOpWithClassVarInvalid)


class MySuperWithClassVarDef(IRDLOperation):
    name = "test.super_has_class_var"

    VAR: ClassVar[str]


class MySubWithClassVarOverload(MySuperWithClassVarDef):
    name = "test.super_has_class_var"

    VAR: ClassVar[str] = "hello_world"


def test_class_var_on_super():
    irdl_op_definition(MySuperWithClassVarDef)
    irdl_op_definition(MySubWithClassVarOverload)
