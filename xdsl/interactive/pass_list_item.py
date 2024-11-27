from textual.widget import Widget
from textual.widgets import ListItem

from xdsl.passes import ModulePass
from xdsl.utils.parse_pipeline import PipelinePassSpec


class PassListItem(ListItem):
    module_pass: type[ModulePass]
    pass_spec: PipelinePassSpec | None

    def __init__(
        self,
        *children: Widget,
        module_pass: type[ModulePass],
        pass_spec: PipelinePassSpec | None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        self.module_pass = module_pass
        self.pass_spec = pass_spec
        super().__init__(
            *children, name=name, id=id, classes=classes, disabled=disabled
        )
