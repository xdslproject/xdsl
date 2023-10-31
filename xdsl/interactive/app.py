from io import StringIO

from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Label, SelectionList, TextArea

from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes


def transform_input(input_text: str, passes: list[type[ModulePass]]) -> str:
    try:
        ctx = MLContext(True)
        for dialect in get_all_dialects():
            ctx.load_dialect(dialect)

        parser = Parser(ctx, input_text)
        module = parser.parse_module()

        pipeline = PipelinePass([p() for p in passes])
        pipeline.apply(ctx, module)

        output_stream = StringIO()
        Printer(output_stream).print(module)
        return output_stream.getvalue()
    except Exception as e:
        return str(e)


# used to prevent being able to change the Output Text Area
class OutputTextArea(TextArea):
    async def _on_key(self, event: events.Key) -> None:
        event.prevent_default()


class InputApp(App[None]):
    """Experimental App."""

    CSS_PATH = "app.tcss"

    text: str

    def __init__(self, text: str | None = None):
        if text is None:
            text = ""
        self.text = text
        super().__init__()

    def compose(self) -> ComposeResult:
        list_of_passes = get_all_passes()
        selections = [(value.name, value) for value in list_of_passes]
        my_selection_list: SelectionList[type[ModulePass]] = SelectionList(
            *selections, id="passes_list"
        )
        yield Label("", id="selected_passes")

        yield my_selection_list
        yield Horizontal(
            TextArea(self.text, id="input"),
            Container(
                OutputTextArea("No output", id="output"),
                id="output-container",
            ),
            id="input_output",
        )
        """yield Horizontal(Button("Generate"))"""

    def selected_module_passes(self) -> list[type[ModulePass]]:
        return self.query_one(SelectionList[type[ModulePass]]).selected

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        new_passes = ",".join(p.name for p in self.selected_module_passes())
        new_label = f"xdsl-opt -p {new_passes}"
        self.query_one(Label).update(new_label)
        self.execute()

    def on_mount(self) -> None:
        self.query_one("#input", TextArea).border_title = "Input"
        self.query_one(SelectionList).border_title = "Choose a pass to be applied."
        """self.query_one(Pretty).border_title = "Selected pass(es)" """
        self.query_one("#output-container").border_title = "Output"
        self.execute()

    # def on_key(self, event: events.Key) -> None:
    #     self.query_one(RichLog).write(event)

    def execute(self, input: TextArea | None = None):
        if input is None:
            input = self.query_one("#input", TextArea)

        passes = self.selected_module_passes()

        input_text = input.text
        output_text = transform_input(input_text, passes)
        output = self.query_one("#output", TextArea)
        output.load_text(output_text)

    @on(TextArea.Changed, "#input")
    def on_input_changed(self, event: TextArea.Changed):
        self.execute(event.text_area)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.exit(str(event.button))


if __name__ == "__main__":
    path = "tests/filecheck/backend/riscv/canonicalize.mlir"
    with open(path) as f:
        text = f.read()
    app = InputApp(text)
    app.run()

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
