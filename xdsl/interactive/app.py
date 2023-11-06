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

from collections.abc import Callable
from io import StringIO

# pyright: ignore[reportMissingTypeStubs, reportGeneralTypeIssues]
from pyclip import copy as pyclip_copy
from rich.style import Style
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Label, SelectionList, TextArea
from textual.widgets.text_area import TextAreaTheme

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes

pyclip_copy: Callable[[str], None] = pyclip_copy


def transform_input(
    input_text: str, passes: list[type[ModulePass]]
) -> builtin.ModuleOp:
    """
    Function that takes the input IR, the list of passes to be applied, and applies the passes to the IR.
    Returns the module (after pass is applied).
    """
    ctx = MLContext(True)
    for dialect in get_all_dialects():
        ctx.load_dialect(dialect)

    parser = Parser(ctx, input_text)
    module = parser.parse_module()

    pipeline = PipelinePass([p() for p in passes])
    pipeline.apply(ctx, module)

    return module


def check_if_pass_changes_IR(
    input: builtin.ModuleOp, ctx: MLContext, one_pass: ModulePass
) -> bool:
    """Used to check if a pass has had an effect on the IR (i.e. has changed the IR)"""
    cloned_module = input.clone()
    cloned_ctx = ctx.clone()
    one_pass.apply(cloned_ctx, cloned_module)
    return not input.is_structurally_equivalent(cloned_module)


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
    text: str

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit_app", "Quit")]

    def __init__(self, text: str | None = None):
        """Initialization function"""
        if text is None:
            text = ""
        self.text = text
        super().__init__()

    def compose(self) -> ComposeResult:
        """
        Creates the required widgets, events, etc.
        Get the list of xDSL passes, add them to an array in "Selection" format (so it can be added to a Selection List)
        and sort the list in alphabetical order.
        """

        my_theme = TextAreaTheme(
            name="my_cool_theme",
            # Basic styles such as background, cursor, selection, gutter, etc...
            base_style=Style(bgcolor="white"),
            # `syntax_styles` is for syntax highlighting.
            # It maps tokens parsed from the document to Rich styles.
            syntax_styles={
                "string": Style(color="red"),
                "comment": Style(color="magenta"),
            },
        )

        list_of_passes = get_all_passes()
        selections = [(value.name, value) for value in list_of_passes]
        selections.sort()
        my_selection_list: SelectionList[type[ModulePass]] = SelectionList(
            *selections, id="passes_selection_list"
        )

        text_area = TextArea(self.text, id="input")
        output_text_area = OutputTextArea("No output", id="output")

        with Horizontal(id="selected_passes_and_list_horizontal"):
            with Horizontal(id="selection_list_and_button"):
                yield my_selection_list
                with VerticalScroll(id="buttons_and_selection_list"):
                    with Horizontal(id="clear_selection_list"):
                        yield Button("Clear Passes", id="clear_selection_list_button")
                    with Horizontal(id="condense_pass_list"):
                        yield Button("Condense", id="condense_pass_list_button")
                    with Horizontal(id="undo_condense"):
                        yield Button("Undo Condense", id="undo_condense_button")
                    with Horizontal(id="copy_query"):
                        yield Button("Copy Query", id="copy_query_button")
            yield ScrollableContainer(
                Label("", id="selected_passes_label"), id="selected_passes"
            )
        with Horizontal(id="input_output"):
            with Vertical(id="input_and_button"):
                yield text_area
                text_area.register_theme(my_theme)
                text_area.theme = "my_cool_theme"
                with Horizontal(id="clear_input"):
                    yield Button("Clear Input", id="clear_input_button")
            with Vertical(id="output_container"):
                yield output_text_area
                output_text_area.register_theme(my_theme)
                output_text_area.theme = "my_cool_theme"
                with Horizontal(id="copy_output"):
                    yield Button("Copy Output", id="copy_output_button")
        yield Footer()

    @property
    def module_passes_selection_list(self) -> SelectionList[type[ModulePass]]:
        return self.query_one(SelectionList[type[ModulePass]])

    def selected_module_passes(self) -> list[type[ModulePass]]:
        """Returns the list of user selected passes"""
        return self.module_passes_selection_list.selected

    @on(SelectionList.SelectedChanged)
    def update_selected_view(self) -> None:
        """
        When the SelectionList (pass options) changes (i.e. a pass was selected or deselected), update the label to show
        the query, and then call the execute() function, which applies the selected passes to the input and displays the output
        """
        new_passes = "\n" + (", " + "\n").join(
            p.name for p in self.selected_module_passes()
        )
        new_label = f"xdsl-opt -p {new_passes}"
        self.query_one(Label).update(new_label)
        self.execute()

    def on_mount(self) -> None:
        """On App Mount, add titles + execute()"""
        self.query_one("#input_and_button").border_title = "Input xDSL IR"
        self.query_one(
            SelectionList
        ).border_title = "Choose a pass or multiple passes to be applied."
        self.query_one("#output_container").border_title = "Output xDSL IR"
        self.query_one("#selected_passes").border_title = "Selected passes/query"
        self.execute()

    def execute(self, input: TextArea | None = None):
        """
        Function that gathers input IR and selected list of passes, calls transform_input() - which returns the module. The output
        stream is then printed and loaded to the output TextArea. (Applies the selected passes to the input IR and displays the output)
        """
        if input is None:
            input = self.query_one("#input", TextArea)

        passes = self.selected_module_passes()

        input_text = input.text
        try:
            module = transform_input(input_text, passes)

            output_stream = StringIO()
            Printer(output_stream).print(module)
            output_text = output_stream.getvalue()
        except Exception as e:
            output_text = str(e)

        output = self.query_one("#output", TextArea)
        output.load_text(output_text)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_quit_app(self) -> None:
        """An action to quit the app."""
        self.exit()

    @on(TextArea.Changed, "#input")
    def on_input_changed(self, event: TextArea.Changed):
        """When the input TextArea changes, call exectue function"""
        self.execute(event.text_area)

    @on(Button.Pressed, "#clear_input_button")
    def on_clear_input_button_pressed(self, event: Button.Pressed) -> None:
        """When the "Clear Input" button is pressed, the input IR TextArea is cleared"""
        input = self.query_one("#input", TextArea)
        input.clear()

    @on(Button.Pressed, "#copy_output_button")
    def on_copy_output_button_pressed(self, event: Button.Pressed) -> None:
        """When the "Copy Output" button is pressed, the output IR TextArea is copied"""
        output = self.query_one("#output", TextArea)
        pyclip_copy(output.text)

    @on(Button.Pressed, "#clear_selection_list_button")
    def on_clear_selection_list_button_pressed(self, event: Button.Pressed) -> None:
        """When the "Clear Passes" button is preseed, the SelectionList is cleared"""
        passes_selection_list = self.query_one("#passes_selection_list", SelectionList)
        selected_passes = self.query_one("#selected_passes_label", Label)
        passes_selection_list.deselect_all()
        self.execute()
        selected_passes.update("")

    @on(Button.Pressed, "#copy_query_button")
    def on_copy_query_button_pressed(self, event: Button.Pressed) -> None:
        """When the "Copy Query" button is preseed, the selected passes/query is copied"""
        selected_passes = "\n" + (", " + "\n").join(
            p.name for p in self.selected_module_passes()
        )
        query = f"xdsl-opt -p {selected_passes}"
        pyclip_copy(query)

    @on(Button.Pressed, "#condense_pass_list_button")
    def on_condense_button_pressed(self, event: Button.Pressed) -> None:
        """
        When the "Condense" button is preseed, the SelectionList is filtered to contain only passes that structurally change the IR
        """
        input = self.query_one("#input", TextArea).text
        if not input:
            return

        passes = get_all_passes()

        try:
            # todo, add parse method that doesnt apply any passes
            module = transform_input(input, [])
        except Exception:
            return

        ctx = MLContext(True)
        for dialect in get_all_dialects():
            ctx.load_dialect(dialect)

        self.module_passes_selection_list.clear_options()
        condense_selections = []

        for p in passes:
            try:
                if check_if_pass_changes_IR(module, ctx, p()):
                    item = (p.name, p)
                    condense_selections.append(item)
            except Exception:
                item = (p.name, p)
                # todo color #todo, add option to add arguments + execute
                condense_selections.append(item)

        condense_selections.sort()
        self.module_passes_selection_list.add_options(condense_selections)

    @on(Button.Pressed, "#undo_condense_button")
    def on_undo_condense_button_pressed(self, event: Button.Pressed) -> None:
        """
        When the "Undo Condense" button is preseed, the SelectionList is returned to its original state
        (i.e. containing all the passes)
        """
        self.module_passes_selection_list.clear_options()
        passes = get_all_passes()
        selections = [(value.name, value) for value in passes]
        selections.sort()
        self.module_passes_selection_list.add_options(selections)


if __name__ == "__main__":
    app = InputApp(None)
    app.run()
