from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, TextArea


class MlirOptPassScreen(Screen[str]):
    CSS_PATH = "mlir_opt_pass_screen.tcss"

    mlir_opt_argument_text_area: TextArea

    def __init__(self):
        self.mlir_opt_argument_text_area = TextArea(
            "", id="mlir_opt_argument_text_area"
        )

        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="container"):
            yield self.mlir_opt_argument_text_area
            with Horizontal(id="cancel_enter_buttons"):
                yield Button("Clear Text", id="clear_input_button")
                yield Button("Enter", id="enter_button")
                yield Button("Cancel", id="quit_screen_button")

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        self.query_one(
            "#mlir_opt_argument_text_area"
        ).border_title = "Provide List of MLIR Passes to apply."

    @on(Button.Pressed, "#quit_screen_button")
    def exit_screen(self, event: Button.Pressed) -> None:
        self.dismiss()

    @on(Button.Pressed, "#clear_input_button")
    def clear_text_area(self, event: Button.Pressed) -> None:
        self.mlir_opt_argument_text_area.clear()

    @on(Button.Pressed, "#enter_button")
    def enter_arguments(self, event: Button.Pressed) -> None:
        self.dismiss(self.mlir_opt_argument_text_area.text)
