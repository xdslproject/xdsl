"""
This app allows you to paste xDSL IR into the Input TextArea, select a pass (or multiple passes) to be applied on the input IR
and subsequently the IR generated by the application of the selected pass(es) is displayed in the Output TextArea. The selected
passes are displayed in the top right "Selecred passes/query" box. The "Condense" button filters the pass selection list to
display passes that change the IR, or require arguments to be executed (and thus "may" change the IR). The "Uncondense" button
returns the selection list to contain all the passes. The "Clear Passes" Button removes the application of the selected passes.
The "Copy Query" Button allows the user to copy the selected passes/query that they have so far selected (i.e. copy the top right
box).

This app is still under construction.
"""

from io import StringIO

from rich.style import Style
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Label, SelectionList, TextArea
from textual.widgets.text_area import TextAreaTheme

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes


class OutputTextArea(TextArea):
    """Used to prevent users from being able to change/alter the Output TextArea"""

    async def _on_key(self, event: events.Key) -> None:
        event.prevent_default()


class InputApp(App[None]):
    """
    Class buildling the Interactive Compilation App. Uses Textual Python to construct reactive variables on the App, Event,
    Widget classes etc. The UI is derived from those values as they change.
    """

    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit_app", "Quit"),
    ]

    current_module: reactive[ModuleOp | Exception | None] = reactive(None)
    """
    Reactive variable used to save the current state of the modified Input TextArea (i.e. is the Output TextArea)
    """

    input_text_area = TextArea(id="input")
    output_text_area = OutputTextArea(id="output")

    list_of_passes = get_all_passes()
    """Contains the list of xDSL passes."""

    # aids in the construction of the seleciton list containing all the passes
    selections = [(value.name, value) for value in list_of_passes]
    selections.sort()
    passes_selection_list: SelectionList[type[ModulePass]] = SelectionList(
        *selections, id="passes_selection_list"
    )

    def compose(self) -> ComposeResult:
        """
        Creates the required widgets, events, etc.
        Get the list of xDSL passes, add them to an array in "Selection" format (so it can be added to a Selection List)
        and sort the list in alphabetical order.
        """

        # defines a theme for the Input/Output TextArea's
        my_theme = TextAreaTheme(
            name="my_theme_design",
            base_style=Style(bgcolor="white"),
            syntax_styles={
                "string": Style(color="red"),
                "comment": Style(color="magenta"),
            },
        )

        # register's the theme for the Input/Output TextArea's
        self.input_text_area.register_theme(my_theme)
        self.input_text_area.theme = "my_theme_design"
        self.output_text_area.register_theme(my_theme)
        self.output_text_area.theme = "my_theme_design"

        # construct the seleciton list containing all the passes and the label displaying the selected passes
        with Horizontal(id="selected_passes_and_list_horizontal"):
            yield self.passes_selection_list
            with ScrollableContainer(id="selected_passes"):
                yield Label("", id="selected_passes_label")

        # construct the input and output TextArea's
        with Horizontal(id="input_output"):
            with Vertical(id="input_container"):
                yield self.input_text_area
            with Vertical(id="output_container"):
                yield self.output_text_area
        yield Footer()

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        """
        When the SelectionList (pass options) changes (i.e. a pass was selected or deselected), update the label to show
        the query, and then call the update_current_module() function, which applies the selected passes to the input and displays the output
        """
        new_passes = "\n" + (", " + "\n").join(
            p.name for p in self.passes_selection_list.selected
        )
        new_label = f"xdsl-opt -p {new_passes}"
        self.query_one(Label).update(new_label)
        self.update_current_module()

    @on(TextArea.Changed, "#input")
    def update_current_module(self) -> None:
        """
        Function called when the Input TextArea is changed. This function parses the Input IR, applies selected passes and updates
        the current_module reactive variable.
        """
        input_text = self.input_text_area.text
        selected_passes = self.passes_selection_list.selected

        try:
            ctx = MLContext(True)
            for dialect in get_all_dialects():
                ctx.load_dialect(dialect)
            parser = Parser(ctx, input_text)
            module = parser.parse_module()

            pipeline = PipelinePass([p() for p in selected_passes])
            pipeline.apply(ctx, module)

            self.current_module = module
        except Exception as e:
            self.current_module = e

    def watch_current_module(self):
        """
        Function called when the current_module reactive variable is updated. This function updates
        the Output TextArea.
        """
        match self.current_module:
            case None:
                output_text = "No input"
            case Exception() as e:
                output_stream = StringIO()
                Printer(output_stream).print(e)
                output_text = output_stream.getvalue()
            case ModuleOp():
                output_stream = StringIO()
                Printer(output_stream).print(self.current_module)
                output_text = output_stream.getvalue()

        self.output_text_area.load_text(output_text)

    def on_mount(self) -> None:
        """On App Mount, add titles"""
        self.query_one("#input_container").border_title = "Input xDSL IR"
        self.query_one("#output_container").border_title = "Output xDSL IR"
        self.query_one(
            "#passes_selection_list"
        ).border_title = "Choose a pass or multiple passes to be applied."
        self.query_one("#selected_passes").border_title = "Selected passes/query"

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_quit_app(self) -> None:
        """An action to quit the app."""
        self.exit()


if __name__ == "__main__":
    app = InputApp(None)
    app.run()
