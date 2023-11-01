from io import StringIO

from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Label, SelectionList, TextArea

from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes


# Function that takes the input IR, the list of passes to be applied, and applies the passes to the IR.
# Returns the output stream (or throws an exception).
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


# Used to prevent users from being able to change/alter the Output TextArea
class OutputTextArea(TextArea):
    async def _on_key(self, event: events.Key) -> None:
        event.prevent_default()


# Class buildling the Experimental Interactive Compilation App.


class InputApp(App[None]):
    CSS_PATH = "app.tcss"
    text: str

    # Initialization function
    def __init__(self, text: str | None = None):
        if text is None:
            text = ""
        self.text = text
        super().__init__()

    # Creates the required widgets, events, etc.
    def compose(self) -> ComposeResult:
        # Get the list of xDSL passes, add them to an array in "Selection" format (so it can be added to a Selection List)
        # and sort the list in alphabetical order.
        list_of_passes = get_all_passes()
        selections = [(value.name, value) for value in list_of_passes]
        selections.sort()
        my_selection_list: SelectionList[type[ModulePass]] = SelectionList(
            *selections, id="passes_selection_list"
        )

        # yield Vertical(
        #     Horizontal(
        #         my_selection_list,  # SelectedList with the pass options
        #         # Label displaying the passes that have been selected
        #         Label("", id="selected_passes"),
        #         id="selected_passes_and_list_horizontal"
        #     ),
        #     Horizontal(
        #         Vertical(
        #             TextArea(self.text, id="input"),  # TextArea with the Input IR
        #             # Horizontal(Button) used to Clear Input TextArea
        #             Horizontal(Button("Clear Input"), id="clear_input"),
        #             id="input_and_button"
        #         ),
        #         Container(
        #             OutputTextArea("No output", id="output"),
        #             id="output_container",
        #         ),  # Container(OutputTextArea) to show Output IR (a TextArea class that has been extended)
        #         id="input_output",
        #     ),
        #     id="overall"
        # )

        # Horizontal(SelectedList, Label) - Container to align the layout of the pass options + displaying selected passes
        with Horizontal(id="selected_passes_and_list_horizontal"):
            yield my_selection_list  # SelectedList with the pass options
            # Label displaying the passes that have been selected
            yield ScrollableContainer(Label(""), id="selected_passes")
        with Horizontal(id="input_output"):
            with Vertical(id="input_and_button"):
                yield TextArea(self.text, id="input")  # TextArea with the Input IR
                with Horizontal(
                    id="clear_input"
                ):  # Horizontal(Button) used to Clear Input TextArea
                    yield Button("Clear Input")
            # Container(OutputTextArea) to show Output IR (a TextArea class that has been extended)
            with Container(id="output_container"):
                yield OutputTextArea("No output", id="output")

    # ORIG
    # yield Horizontal(
    #     my_selection_list,  # SelectedList with the pass options
    #     # Label displaying the passes that have been selected
    #     Label("", id="selected_passes"),
    #     id="selected_passes_and_list_horizontal"
    # )

    # yield Horizontal(
    #     Vertical(
    #         TextArea(self.text, id="input"),  # TextArea with the Input IR
    #         # Horizontal(Button) used to Clear Input TextArea
    #         Horizontal(Button("Clear Input"), id="clear_input"),
    #         id="input_and_button"
    #     ),
    #     Container(
    #         OutputTextArea("No output", id="output"),
    #         id="output_container",
    #     ),  # Container(OutputTextArea) to show Output IR (a TextArea class that has been extended)
    #     id="input_output",
    # )

    # Returns the list of user selected passes

    def selected_module_passes(self) -> list[type[ModulePass]]:
        return self.query_one(SelectionList[type[ModulePass]]).selected

    # When the SelectionList (pass options) changes (i.e. a pass was selected or deselected), update the label to show
    # the query, and then call the execute() function, which applies the selected passes to the input and displays the output
    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        new_passes = "\n" + (", " + "\n").join(
            p.name for p in self.selected_module_passes()
        )
        new_label = f"xdsl-opt -p {new_passes}"
        self.query_one(Label).update(new_label)
        self.execute()

    # On App Mount, add titles + execute()

    def on_mount(self) -> None:
        self.query_one("#input_and_button").border_title = "Input xDSL IR"
        self.query_one(
            SelectionList
        ).border_title = "Choose a pass or multiple passes to be applied."
        self.query_one("#output_container").border_title = "Output xDSL IR"
        self.execute()

    # Function that gathers input IR and selected list of passes, calls transform_input() which returns the output stream,
    # and loads it to the output TextArea. (Applies the selected passes to the input IR and displays the output)
    def execute(self, input: TextArea | None = None):
        if input is None:
            input = self.query_one("#input", TextArea)

        passes = self.selected_module_passes()

        input_text = input.text
        output_text = transform_input(input_text, passes)
        output = self.query_one("#output", TextArea)
        output.load_text(output_text)

    # When the input TextArea changes, call exectue function
    @on(TextArea.Changed, "#input")
    def on_input_changed(self, event: TextArea.Changed):
        self.execute(event.text_area)

    # When the "Clear Input" button is pressed, the input IR TextArea is cleared
    def on_button_pressed(self, event: Button.Pressed) -> None:
        input = self.query_one("#input", TextArea)
        input.clear()


if __name__ == "__main__":
    path = "tests/filecheck/backend/riscv/canonicalize.mlir"
    with open(path) as f:
        text = f.read()
    app = InputApp(text)
    app.run()
