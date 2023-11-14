"""
An interactive command-line tool to explore compilation pipeline construction.

Execute `xdsl-gui` in your terminal to run it.

Run `terminal -m xdsl.interactive.app:InputApp --dev` to run in development mode. Please

be sure to install `textual-dev` to run this command.
"""

from io import StringIO

from rich.style import Style
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Footer, Label, SelectionList, TextArea
from textual.widgets.text_area import TextAreaTheme

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes

from ._pasteboard import pyclip_copy

ALL_PASSES = tuple(get_all_passes())
"""Contains the list of xDSL passes."""


class OutputTextArea(TextArea):
    """Used to prevent users from being able to change/alter the Output TextArea"""

    async def _on_key(self, event: events.Key) -> None:
        event.prevent_default()


class InputApp(App[None]):
    """
    Interactive application for constructing compiler pipelines.
    """

    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit_app", "Quit"),
    ]

    # defines a theme for the Input/Output TextArea's
    _DEFAULT_THEME = TextAreaTheme(
        name="my_theme_design",
        base_style=Style(bgcolor="white"),
        syntax_styles={
            "string": Style(color="red"),
            "comment": Style(color="magenta"),
        },
    )

    current_module = reactive[ModuleOp | Exception | None](None)
    """
    Reactive variable used to save the current state of the modified Input TextArea
    (i.e. is the Output TextArea)
    """

    input_text_area: TextArea
    output_text_area: OutputTextArea
    passes_selection_list: SelectionList[type[ModulePass]]

    selected_query_label: Label
    """Display selected passes"""

    def __init__(self):
        self.input_text_area = TextArea(id="input")
        self.output_text_area = OutputTextArea(id="output")
        self.passes_selection_list = SelectionList(id="passes_selection_list")
        self.selected_query_label = Label("", id="selected_passes_label")
        super().__init__()

    def compose(self) -> ComposeResult:
        """
        Creates the required widgets, events, etc.
        Get the list of xDSL passes, add them to an array in "Selection" format (so it
        can be added to a Selection List)
        and sort the list in alphabetical order.
        """
        with Horizontal(id="top_container"):
            yield self.passes_selection_list
            with Horizontal(id="button_and_selected_horziontal"):
                with VerticalScroll(id="buttons_and_selection_list"):
                    yield Button("Clear Passes", id="clear_selection_list_button")
                    yield Button("Copy Query", id="copy_query_button")
                with ScrollableContainer(id="selected_passes"):
                    yield self.selected_query_label
        with Horizontal(id="bottom_container")
            with Vertical(id="input_container"):
                yield self.input_text_area
                yield Button("Clear Input", id="clear_input_button")
            with Vertical(id="output_container"):
                yield self.output_text_area
                yield Button("Copy Output", id="copy_output_button")
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

    @on(SelectionList.SelectedChanged)
    @on(TextArea.Changed, "#input")
    def update_current_module(self) -> None:
        """
        Function called when the Input TextArea is changed or a pass is selected/
        unselected. This function parses the Input IR, applies selected passes and
        updates the Output TextArea.
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
        Function called when the current_module reactive variable is updated. This
        function updates the Output TextArea.
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
        """Configure widgets in this application before it is first shown."""

        # register's the theme for the Input/Output TextArea's
        self.input_text_area.register_theme(InputApp._DEFAULT_THEME)
        self.output_text_area.register_theme(InputApp._DEFAULT_THEME)
        self.input_text_area.theme = "my_theme_design"
        self.output_text_area.theme = "my_theme_design"

        self.query_one("#input_container").border_title = "Input xDSL IR"
        self.query_one("#output_container").border_title = "Output xDSL IR"
        self.query_one(
            "#passes_selection_list"
        ).border_title = "Choose a pass or multiple passes to be applied."
        self.query_one("#selected_passes").border_title = "Selected passes/query"
        # aids in the construction of the seleciton list containing all the passes
        selections = sorted((value.name, value) for value in ALL_PASSES)

        # type error due to Textual Bug requires pyright ignore
        # Link to issue: https://github.com/xdslproject/xdsl/issues/1777
        self.passes_selection_list.add_options(  # pyright: ignore[reportUnknownMemberType]
            selections
        )

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_quit_app(self) -> None:
        """An action to quit the app."""
        self.exit()

    @on(Button.Pressed, "#clear_input_button")
    def clear_input(self, event: Button.Pressed) -> None:
        """Input TextArea is cleared when "Clear Input" button is pressed"""
        self.input_text_area.clear()

    @on(Button.Pressed, "#copy_output_button")
    def copy_output(self, event: Button.Pressed) -> None:
        """When the "Copy Output" button is pressed, the output IR TextArea is copied"""
        pyclip_copy(self.output_text_area.text)

    @on(Button.Pressed, "#clear_selection_list_button")
    def clear_selection_list(self, event: Button.Pressed) -> None:
        """When the "Clear Passes" button is preseed, the SelectionList is cleared"""
        self.passes_selection_list.deselect_all()

    @on(Button.Pressed, "#copy_query_button")
    def copy_query(self, event: Button.Pressed) -> None:
        """When the "Copy Query" button is preseed, the selected passes/query is copied"""
        selected_passes = "\n" + (", " + "\n").join(
            p.name for p in self.passes_selection_list.selected
        )
        query = f"xdsl-opt -p {selected_passes}"
        pyclip_copy(query)


def main():
    return InputApp().run()


if __name__ == "__main__":
    main()
