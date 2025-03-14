from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.reactive import Reactive
from textual.screen import Screen
from textual.widgets import Button, TextArea

from xdsl.passes import ModulePass, get_pass_option_infos
from xdsl.utils.exceptions import PassPipelineParseError
from xdsl.utils.parse_pipeline import parse_pipeline


class AddArguments(Screen[ModulePass | None]):
    """
    Screen called when selected pass has arguments requiring user input.
    """

    CSS_PATH = "add_arguments_screen.tcss"

    selected_pass_type: type[ModulePass]
    selected_pass_value = Reactive[ModulePass | None](None)
    argument_text_area: TextArea

    def __init__(self, selected_pass_type: type[ModulePass]):
        self.selected_pass_type = selected_pass_type
        self.argument_text_area = TextArea(
            " ".join(
                f"{n}={t if d is None else d}"
                for n, t, d in get_pass_option_infos(selected_pass_type)
            ),
            id="argument_text_area",
        )
        self.enter_button = Button("Enter", id="enter_button")

        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="container"):
            yield self.argument_text_area
            with Horizontal(id="cancel_enter_buttons"):
                yield Button("Clear Text", id="clear_input_screen_button")
                yield self.enter_button
                yield Button("Cancel", id="quit_screen_button")

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        self.query_one(
            "#argument_text_area"
        ).border_title = "Provide arguments to apply to selected pass."
        # Initialize parsed pass
        self.update_selected_pass_value()
        # Initialize enter button
        self.watch_selected_pass_value()

    @on(TextArea.Changed, "#argument_text_area")
    def update_selected_pass_value(self) -> None:
        concatenated_arg_val = self.argument_text_area.text

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
        self.argument_text_area.load_text("")

    @on(Button.Pressed, "#enter_button")
    def enter_arguments(self, event: Button.Pressed) -> None:
        assert self.selected_pass_value is not None
        self.dismiss(self.selected_pass_value)
