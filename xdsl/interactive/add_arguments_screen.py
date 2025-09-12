from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import Reactive
from textual.screen import Screen
from textual.widgets import Button, Input, Label

from xdsl.passes import ModulePass, PassOptionInfo, get_pass_option_infos
from xdsl.utils.exceptions import PassPipelineParseError
from xdsl.utils.parse_pipeline import parse_pipeline


class AddArguments(Screen[ModulePass | None]):
    """
    Screen called when selected pass has arguments requiring user input.
    """

    CSS_PATH = "add_arguments_screen.tcss"

    selected_pass_type: type[ModulePass]
    selected_pass_value = Reactive[ModulePass | None](None)
    argument_tuple: tuple[PassOptionInfo, ...]

    def __init__(self, selected_pass_type: type[ModulePass]):
        self.selected_pass_type = selected_pass_type
        self.argument_tuple = tuple(
            PassOptionInfo(name=n, expected_type=t, default_value=d)
            for n, t, d in get_pass_option_infos(selected_pass_type)
        )
        self.enter_button = Button("Enter", id="enter_button")

        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="top_level"):
            with Vertical(id="argument_container"):
                for arg in self.argument_tuple:
                    with Horizontal(id=f"argument_row_{arg.name}"):
                        yield Label(
                            f"{arg.name}",
                            id=f"arg_name_{arg.name}",
                        )
            with Vertical(id="input_container"):
                for arg in self.argument_tuple:
                    default_str = (
                        f" [default: {arg.default_value}]"
                        if arg.default_value is not None
                        else ""
                    )
                    yield Input(
                        placeholder=f"Enter {arg.expected_type} value here.{default_str}",
                        id=f"input_{arg.name}",
                    )
        with Horizontal(id="cancel_enter_buttons"):
            yield Button("Clear Text", id="clear_input_screen_button")
            yield self.enter_button
            yield Button("Cancel", id="quit_screen_button")

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        self.query_one(
            "#top_level"
        ).border_title = "Provide arguments to apply to selected pass."
        self.query_one(
            "#argument_container"
        ).border_title = f"Arguments for {self.selected_pass_type.name} pass"
        self.query_one("#input_container").border_title = "Input Values"

        # Initialize parsed pass
        self.update_selected_pass_value()
        # Initialize enter button
        self.watch_selected_pass_value()

    @on(Input.Changed)
    def update_selected_pass_value(self) -> None:
        concatenated_arg_val = ""
        for arg in self.argument_tuple:
            input_widget = self.query_one(f"#input_{arg.name}", Input)
            if concatenated_arg_val != "":
                concatenated_arg_val += " "
            concatenated_arg_val += arg.name + "=" + str(input_widget.value)

        try:
            parsed_spec = list(
                parse_pipeline(
                    f"{self.selected_pass_type.name}{{{concatenated_arg_val}}}"
                )
            )[0]
        except PassPipelineParseError:
            self.selected_pass_value = None
            return

        missing_fields = self.selected_pass_type.required_fields().difference(
            parsed_spec.args.keys()
        )

        if missing_fields:
            self.selected_pass_value = None
            return

        try:
            self.selected_pass_value = self.selected_pass_type.from_pass_spec(
                parsed_spec
            )
        except ValueError:
            self.selected_pass_value = None

    def watch_selected_pass_value(self):
        self.enter_button.disabled = self.selected_pass_value is None

    @on(Button.Pressed, "#quit_screen_button")
    def exit_screen(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#clear_input_screen_button")
    def clear_text_area(self, event: Button.Pressed) -> None:
        for arg in self.argument_tuple:
            input_widget = self.query_one(f"#input_{arg.name}", Input)
            input_widget.clear()

    @on(Button.Pressed, "#enter_button")
    def enter_arguments(self, event: Button.Pressed) -> None:
        assert self.selected_pass_value is not None
        self.dismiss(self.selected_pass_value)
