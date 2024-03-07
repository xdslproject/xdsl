from xdsl.ir import Attribute, SSAValue


class Ctx:
    type_by_name: dict[str, Attribute]
    value_by_name: dict[str, SSAValue]

    def __init__(self):
        self.type_by_name = {}
        self.value_by_name = {}
