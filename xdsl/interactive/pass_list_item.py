from textual.widget import Widget
from textual.widgets import ListItem

from xdsl.passes import ModulePass


class PassListItem(ListItem):
    module_pass: type[ModulePass] | ModulePass

    def __init__(
        self,
        *children: Widget,
        module_pass: type[ModulePass] | ModulePass,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        self.module_pass = module_pass
        super().__init__(
            *children, name=name, id=id, classes=classes, disabled=disabled
        )
