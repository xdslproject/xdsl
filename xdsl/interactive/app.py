from io import StringIO

from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.events import Mount
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    Pretty,
    RichLog,
    SelectionList,
    Static,
    TextArea,
)

from xdsl.ir.core import MLContext
from xdsl.parser.core import Parser
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects


def transform_input(input_text: str) -> str:
    try:
        ctx = MLContext(True)
        for dialect in get_all_dialects():
            ctx.register_dialect(dialect)

        parser = Parser(ctx, input_text)
        module = parser.parse_module()

        output_stream = StringIO()
        Printer(output_stream).print(module)
        return output_stream.getvalue()
    except Exception as e:
        return str(e)


"""
def run_terminal_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True,
                                text=True, capture_output=True)
        # Output from the terminal command
        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        # If the command exits with a non-zero status code, it will raise an exception.
        print(f"Error: {e}")
        return None
"""


class InputApp(App[None]):
    """Experimental App."""

    CSS_PATH = "app.tcss"

    text: str
    x = 0

    def __init__(self, text: str | None = None):
        if text is None:
            text = ""
        self.text = text
        super().__init__()

    def compose(self) -> ComposeResult:
        yield TextArea(self.text, id="input")
        yield Container(VerticalScroll(Label(self.text, id="output")))
        yield Static("Tree", id="sidebar")
        yield Header()
        yield SelectionList[str](
            ("Pass 1", "-canonicalize"),
            ("Pass 2", "-print-ir"),
            ("Pass 3", "-cse"),
            ("Pass 4", "-remove-dead-values"),
            id="passes",
        )
        yield Pretty([], id="passes_list")
        yield Footer()
        yield Horizontal(Button("Generate"))

    @on(Mount)
    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        self.query_one(Pretty).update(self.query_one(SelectionList).selected)

    def on_mount(self) -> None:
        self.query_one("#input", TextArea).border_title = "Input"
        self.query_one(SelectionList).border_title = "Choose a pass to be performed."
        self.query_one(Pretty).border_title = "Selected pass(es)"
        self.query_one("#output", Label).border_title = "Output"

    def on_key(self, event: events.Key) -> None:
        self.query_one(RichLog).write(event)

    def on_text_area_changed(self, event: TextArea.Changed):
        input_text = event.text_area.text
        output_text = transform_input(input_text)

        output = self.query_one("#output", Label)
        output.update(output_text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.exit(str(event.button))

    """
    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Define a variable to keep track of the current file number
        file_number = 1

        # Construct the file name based on the current file number
        file_path = f"file{file_number}.mlir"

        # Increment the file number for the next call
        file_number += 1

        text_input = self.query_one("#input", TextArea).text
        text_passes = self.query_one("#passes", SelectionList).selected

        # Open and write to the file
        with open(file_path, "w") as file:
            file.write(text_input)

        # Replace this with the command you want to run
        command_to_run = "mlir-opt " + \
            " [" + ", ".join(text_passes) + "] " + "<" + file_path + ">"
        output = run_terminal_command(command_to_run)

        output_lbl = self.query_one("#output", Label)
        output_lbl.update(output)  # pyright: ignore[reportGeneralTypeIssues]
    """


if __name__ == "__main__":
    path = "bla.mlir"
    with open(path) as f:
        text = f.read()
    app = InputApp(text)
    app.run()
