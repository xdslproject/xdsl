from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, TextArea


class AddArguments(Screen[str | None]):
    """
    Screen called when selected pass has arguments requiring user input.
    """

    CSS_PATH = "add_arguments_screen.tcss"

    argument_text_area: TextArea

    def __init__(self, parameter_one: TextArea):
        self.argument_text_area = parameter_one
        self.input_text_area = TextArea(id="input")

        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="container"):
            yield self.argument_text_area
            with Horizontal(id="cancel_enter_buttons"):
                yield Button("Clear Text", id="clear_input_screen_button")
                yield Button("Enter", id="enter_button")
                yield Button("Cancel", id="quit_screen_button")

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        self.query_one(
            "#argument_text_area"
        ).border_title = "Provide arguments to apply to selected pass."

    @on(Button.Pressed, "#quit_screen_button")
    def exit_screen(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#clear_input_screen_button")
    def clear_text_area(self, event: Button.Pressed) -> None:
        self.argument_text_area.load_text("")

    @on(Button.Pressed, "#enter_button")
    def enter_arguments(self, event: Button.Pressed) -> None:
        self.dismiss(self.argument_text_area.text)
