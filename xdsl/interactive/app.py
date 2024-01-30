"""
An interactive command-line tool to explore compilation pipeline construction.

Execute `xdsl-gui` in your terminal to run it.

Run `textual run xdsl.interactive.app:InputApp --dev` to run in development mode. Please
be sure to install `textual-dev` to run this command.
"""

import argparse
import os
from collections.abc import Callable
from dataclasses import fields
from io import StringIO
from typing import Any, ClassVar

from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Label,
    ListItem,
    ListView,
    TextArea,
)

from xdsl.dialects import builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.add_arguments_screen import AddArguments
from xdsl.interactive.load_file_screen import LoadFile
from xdsl.interactive.pass_metrics import (
    count_number_of_operations,
    get_diff_operation_count,
)
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass, get_pass_argument_names_and_types
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects, get_all_passes
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.exceptions import PassPipelineParseError
from xdsl.utils.parse_pipeline import PipelinePassSpec, parse_pipeline

from ._pasteboard import pyclip_copy

ALL_PASSES = tuple(sorted((p_name, p()) for (p_name, p) in get_all_passes().items()))
"""Contains the list of xDSL passes."""


def condensed_pass_list(input: builtin.ModuleOp) -> tuple[type[ModulePass], ...]:
    """Returns a tuple of passes (pass name and pass instance) that modify the IR."""

    ctx = MLContext(True)

    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    selections: list[type[ModulePass]] = []
    for _, value in ALL_PASSES:
        if value is MLIROptPass:
            # Always keep MLIROptPass as an option in condensed list
            selections.append(value)
            continue
        try:
            cloned_module = input.clone()
            cloned_ctx = ctx.clone()
            value().apply(cloned_ctx, cloned_module)
            if input.is_structurally_equivalent(cloned_module):
                continue
        except Exception:
            pass
        selections.append(value)

    return tuple(selections)


class OutputTextArea(TextArea):
    """Used to prevent users from being able to alter the Output TextArea."""

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

    SCREENS: ClassVar[dict[str, type[Screen[Any]] | Callable[[], Screen[Any]]]] = {
        "add_arguments_screen": AddArguments,
        "load_file": LoadFile,
    }
    """
    A dictionary that maps names on to Screen objects.
    """

    INITIAL_IR_TEXT = """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """

    current_module = reactive[ModuleOp | Exception | None](None)
    """
    Reactive variable used to save the current state of the modified Input TextArea
    (i.e. is the Output TextArea).
    """
    pass_pipeline = reactive(tuple[tuple[type[ModulePass], PipelinePassSpec], ...])
    """Reactive variable that saves the list of selected passes."""

    condense_mode = reactive(False, always_update=True)
    """Reactive boolean."""
    available_pass_list = reactive(tuple[type[ModulePass], ...])
    """
    Reactive variable that saves the list of passes that have an effect on
    current_module.
    """

    input_text_area: TextArea
    """Input TextArea."""
    output_text_area: OutputTextArea
    """Output TextArea."""
    selected_query_label: Label
    """Display selected passes."""
    passes_list_view: ListView
    """ListView displaying the passes available to apply."""

    input_operation_count_tuple = reactive(tuple[tuple[str, int], ...])
    """
    Saves the operation name and count of the input text area in a reactive tuple of
    tuples.
    """
    diff_operation_count_tuple = reactive(tuple[tuple[str, int, str], ...])
    """
    Saves the diff of the input_operation_count_tuple and the output_operation_count_tuple
    in a reactive tuple of tuples.
    """

    input_operation_count_datatable: DataTable[str | int]
    """DataTable displaying the operation names and counts of the input text area."""
    diff_operation_count_datatable: DataTable[str | int]
    """
    DataTable displaying the diff of operation names and counts of the input and output
    text areas.
    """

    pre_loaded_input_text: str
    pre_loaded_pass_pipeline: tuple[tuple[type[ModulePass], PipelinePassSpec], ...]

    def __init__(
        self,
        input_text: str | None = None,
        pass_pipeline: tuple[tuple[type[ModulePass], PipelinePassSpec], ...] = (),
    ):
        self.input_text_area = TextArea(id="input")
        self.output_text_area = OutputTextArea(id="output")
        self.passes_list_view = ListView(id="passes_list_view")
        self.selected_query_label = Label("", id="selected_passes_label")
        self.input_operation_count_datatable = DataTable(
            id="input_operation_count_datatable"
        )
        self.diff_operation_count_datatable = DataTable(
            id="diff_operation_count_datatable"
        )

        if input_text is None:
            self.pre_loaded_input_text = InputApp.INITIAL_IR_TEXT
        else:
            self.pre_loaded_input_text = input_text

        self.pre_loaded_pass_pipeline = pass_pipeline

        super().__init__()

    def compose(self) -> ComposeResult:
        """
        Creates the required widgets, events, etc.
        """

        with Horizontal(id="top_container"):
            yield self.passes_list_view
            with Horizontal(id="button_and_selected_horziontal"):
                with ScrollableContainer(id="buttons"):
                    yield Button("Copy Query", id="copy_query_button")
                    yield Button("Clear Passes", id="clear_passes_button")
                    yield Button("Condense", id="condense_button")
                    yield Button("Uncondense", id="uncondense_button")
                    yield Button("Remove Last Pass", id="remove_last_pass_button")
                    yield Button(
                        "Show Operation Count", id="show_operation_count_button"
                    )
                    yield Button(
                        "Remove Operation Count", id="remove_operation_count_button"
                    )
                with ScrollableContainer(id="selected_passes"):
                    yield self.selected_query_label
        with Horizontal(id="bottom_container"):
            with Horizontal(id="input_horizontal_container"):
                with Vertical(id="input_container"):
                    yield self.input_text_area
                    with Horizontal(id="input_horizontal"):
                        yield Button("Clear Input", id="clear_input_button")
                        yield Button("Load File", id="load_file_button")
                with ScrollableContainer(id="input_ops_container"):
                    yield self.input_operation_count_datatable

            with Horizontal(id="output_horizontal_container"):
                with Vertical(id="output_container"):
                    yield self.output_text_area
                    yield Button("Copy Output", id="copy_output_button")
                with ScrollableContainer(id="output_ops_container"):
                    yield self.diff_operation_count_datatable
        yield Footer()

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        # Registers the theme for the Input/Output TextAreas
        self.input_text_area.theme = "vscode_dark"
        self.output_text_area.theme = "vscode_dark"

        # add titles for various widgets
        self.query_one("#input_container").border_title = "Input xDSL IR"
        self.query_one("#output_container").border_title = "Output xDSL IR"
        self.query_one(
            "#passes_list_view"
        ).border_title = "Choose a pass or multiple passes to be applied."
        self.query_one("#selected_passes").border_title = "Selected passes/query"

        # initialize ListView to contain the pass options
        for n, _ in ALL_PASSES:
            self.passes_list_view.append(ListItem(Label(n), name=n))

        # initialize GUI with either specified input text or default example
        self.input_text_area.load_text(self.pre_loaded_input_text)

        # initialize DataTable with column names
        self.input_operation_count_datatable.add_columns("Operation", "Count")
        self.input_operation_count_datatable.zebra_stripes = True

        self.diff_operation_count_datatable.add_columns("Operation", "Count", "Diff")
        self.diff_operation_count_datatable.zebra_stripes = True

        # initialize GUI with specified pass pipeline
        self.pass_pipeline = self.pre_loaded_pass_pipeline

    def compute_available_pass_list(self) -> tuple[type[ModulePass], ...]:
        """
        When any reactive variable is modified, this function (re-)computes the
        available_pass_list variable.
        """
        match self.current_module:
            case None:
                return tuple(p for _, p in ALL_PASSES)
            case Exception():
                return ()
            case ModuleOp():
                if self.condense_mode:
                    return condensed_pass_list(self.current_module)
                else:
                    return tuple(p for _, p in ALL_PASSES)

    def watch_available_pass_list(
        self,
        old_pass_list: tuple[type[ModulePass], ...],
        new_pass_list: tuple[type[ModulePass], ...],
    ) -> None:
        """
        Function called when the reactive variable available_pass_list changes - updates
        the ListView to display the latest pass options.
        """
        if old_pass_list != new_pass_list:
            self.passes_list_view.clear()
            for value in new_pass_list:
                self.passes_list_view.append(
                    ListItem(Label(value.name), name=value.name)
                )

    def get_pass_arguments(self, selected_pass_value: type[ModulePass]) -> None:
        """
        This function facilitates user input of pass concatenated_arg_val by navigating
        to the AddArguments screen, and subsequently parses the returned string upon
        screen dismissal and appends the pass to the pass_pipeline variable.
        """

        def add_pass_with_arguments_to_pass_pipeline(concatenated_arg_val: str) -> None:
            """
            Called when AddArguments Screen is dismissed. This function attempts to parse
            the returned string, and if successful, adds it to the pass_pipeline variable.
            In case of parsing failure, the AddArguments Screen is pushed, revealing the
            Parse Error.
            """
            try:
                new_pass_with_arguments = list(
                    parse_pipeline(
                        f"{selected_pass_value.name}{{{concatenated_arg_val}}}"
                    )
                )[0]
                self.pass_pipeline = (
                    *self.pass_pipeline,
                    (selected_pass_value, new_pass_with_arguments),
                )

            except PassPipelineParseError as e:
                res = f"PassPipelineParseError: {e}"
                screen = AddArguments(TextArea(res, id="argument_text_area"))
                self.push_screen(screen, add_pass_with_arguments_to_pass_pipeline)

        # if selected_pass_value has arguments, push screen
        if fields(selected_pass_value):
            # generates a string containing the concatenated_arg_val and types of the selected pass and initializes the AddArguments Screen to contain the string
            self.push_screen(
                AddArguments(
                    TextArea(
                        get_pass_argument_names_and_types(selected_pass_value),
                        id="argument_text_area",
                    )
                ),
                add_pass_with_arguments_to_pass_pipeline,
            )
        else:
            # add the selected pass to pass_pipeline
            self.pass_pipeline = (
                *self.pass_pipeline,
                (selected_pass_value, selected_pass_value().pipeline_pass_spec()),
            )

    @on(ListView.Selected)
    def update_pass_pipeline(self, event: ListView.Selected) -> None:
        """
        When a new selection is made, the reactive variable storing the list of selected
        passes is updated.
        """
        selected_pass = event.item.name
        for pass_name, pass_value in ALL_PASSES:
            if pass_name == selected_pass:
                # check if pass has arguments
                self.get_pass_arguments(pass_value)

    def watch_pass_pipeline(self) -> None:
        """
        Function called when the reactive variable pass_pipeline changes - updates the
        label to display the respective generated query in the Label.
        """
        self.selected_query_label.update(self.get_query_string())
        self.update_current_module()

    @on(TextArea.Changed, "#input")
    def update_current_module(self) -> None:
        """
        Function to parse the input and to apply the list of selected passes to it.
        """
        input_text = self.input_text_area.text
        if (input_text) == "":
            self.current_module = None
            self.current_condensed_pass_list = ()
            self.update_input_operation_count_tuple(ModuleOp([], None))
            return
        try:
            ctx = MLContext(True)
            for dialect_name, dialect_factory in get_all_dialects().items():
                ctx.register_dialect(dialect_name, dialect_factory)
            parser = Parser(ctx, input_text)
            module = parser.parse_module()
            self.update_input_operation_count_tuple(module)
            pipeline = PipelinePass(
                passes=[
                    module_pass.from_pass_spec(pipeline_pass_spec)
                    for module_pass, pipeline_pass_spec in self.pass_pipeline
                ]
            )
            pipeline.apply(ctx, module)
            self.current_module = module
        except Exception as e:
            self.current_module = e
            self.update_input_operation_count_tuple(ModuleOp([], None))

    def watch_current_module(self):
        """
        Function called when the reactive variable current_module changes - updates the
        Output TextArea.
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
        self.update_operation_count_diff_tuple()

    def get_query_string(self) -> str:
        """
        Function returning a string containing the textual description of the pass
        pipeline generated thus far.
        """
        query = ""
        if self.pass_pipeline != ():
            query += "'"
            query += ",".join(
                str(pipeline_pass_spec) for _, pipeline_pass_spec in self.pass_pipeline
            )
            query += "'"
        return f"xdsl-opt -p {query}"

    def update_input_operation_count_tuple(self, input_module: ModuleOp) -> None:
        """
        Function that updates the input_operation_datatable to display the operation
        names and counts in the input text area.
        """
        # sort tuples alphabetically by operation name
        self.input_operation_count_tuple = tuple(
            sorted(count_number_of_operations(input_module).items())
        )

    def watch_input_operation_count_tuple(self) -> None:
        """
        Function called when the reactive variable input_operation_count_tuple changes - updates the
        Input DataTable.
        """
        # clear datatable and add input_operation_count_tuple to DataTable
        self.input_operation_count_datatable.clear()
        self.input_operation_count_datatable.add_rows(self.input_operation_count_tuple)
        self.update_operation_count_diff_tuple()

    def update_operation_count_diff_tuple(self) -> None:
        """
        Function that updates the diff_operation_count_tuple to calculate the diff
        of the input and output operation counts.
        """
        match self.current_module:
            case None:
                output_operation_count_tuple = ()
            case Exception():
                output_operation_count_tuple = ()
            case ModuleOp():
                # sort tuples alphabetically by operation name
                output_operation_count_tuple = tuple(
                    (k, v)
                    for (k, v) in sorted(
                        count_number_of_operations(self.current_module).items()
                    )
                )
        self.diff_operation_count_tuple = get_diff_operation_count(
            self.input_operation_count_tuple, output_operation_count_tuple
        )

    def watch_diff_operation_count_tuple(self) -> None:
        """
        Function called when the reactive variable diff_operation_count_tuple changes
        - updates the Output DataTable.
        """
        self.diff_operation_count_datatable.clear()
        self.diff_operation_count_datatable.add_rows(self.diff_operation_count_tuple)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark

    def action_quit_app(self) -> None:
        """An action to quit the app."""
        self.exit()

    @on(Button.Pressed, "#clear_input_button")
    def clear_input(self, event: Button.Pressed) -> None:
        """Input TextArea is cleared when "Clear Input" button is pressed."""
        self.input_text_area.clear()

    @on(Button.Pressed, "#copy_output_button")
    def copy_output(self, event: Button.Pressed) -> None:
        """Output TextArea is copied when "Copy Output" button is pressed."""
        pyclip_copy(self.output_text_area.text)

    @on(Button.Pressed, "#copy_query_button")
    def copy_query(self, event: Button.Pressed) -> None:
        """Selected passes/query Label is copied when "Copy Query" button is pressed."""
        pyclip_copy(self.get_query_string())

    @on(Button.Pressed, "#clear_passes_button")
    def clear_passes(self, event: Button.Pressed) -> None:
        """Selected passes cleared when "Clear Passes" button is pressed."""
        self.pass_pipeline = ()

    @on(Button.Pressed, "#condense_button")
    def condense(self, event: Button.Pressed) -> None:
        """
        Displayed passes are filtered to display only those passes that have an affect
        on current_module when "Condense" Button is pressed.
        """
        self.condense_mode = True
        self.add_class("condensed")

    @on(Button.Pressed, "#uncondense_button")
    def uncondense(self, event: Button.Pressed) -> None:
        """
        Displayed passes are filtered to display all available passes when "Uncondense"
        Button is pressed.
        """
        self.condense_mode = False
        self.remove_class("condensed")

    @on(Button.Pressed, "#show_operation_count_button")
    def show_operation_count_button(self, event: Button.Pressed) -> None:
        """Operation Count is displayed when "Show Operation Count" button is pressed."""
        self.add_class("operation_count_shown")

    @on(Button.Pressed, "#remove_operation_count_button")
    def remove_operation_count_button(self, event: Button.Pressed) -> None:
        """Operation Count is removed when "Remove Operation Count" button is pressed."""
        self.remove_class("operation_count_shown")

    @on(Button.Pressed, "#remove_last_pass_button")
    def remove_last_pass(self, event: Button.Pressed) -> None:
        """Last selected pass removed when "Remove Last Pass" button is pressed."""
        self.pass_pipeline = self.pass_pipeline[:-1]

    @on(Button.Pressed, "#load_file_button")
    def load_file(self, event: Button.Pressed) -> None:
        """
        Pushes screen displaying DirectoryTree widget when "Load File" button is pressed.
        """

        def check_load_file(file_path: str) -> None:
            """
            Called when LoadFile is dismissed. Loads selected file into
            input_text_area.
            """
            # Clear Input TextArea and Pass Pipeline
            self.pass_pipeline = ()
            self.input_text_area.clear()

            try:
                if os.path.exists(file_path):
                    # Open the file and read its contents
                    with open(file_path) as file:
                        file_contents = file.read()
                        self.input_text_area.load_text(file_contents)
                else:
                    self.input_text_area.load_text(
                        f"The file '{file_path}' does not exist."
                    )
            except Exception as e:
                self.input_text_area.load_text(str(e))

        self.push_screen("load_file", check_load_file)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "input_file", type=str, nargs="?", help="path to input file"
    )

    available_passes = ",".join([name for name in get_all_passes()])
    arg_parser.add_argument(
        "-p",
        "--passes",
        required=False,
        help="Delimited list of passes." f" Available passes are: {available_passes}",
        type=str,
        default="",
    )
    args = arg_parser.parse_args()

    file_path = args.input_file
    if file_path is not None:
        # Open the file and read its contents
        with open(file_path) as file:
            file_contents = file.read()
    else:
        file_contents = None

    pass_spec_pipeline = list(parse_pipeline(args.passes))
    pass_list = get_all_passes()
    pipeline = tuple(PipelinePass.build_pipeline_tuples(pass_list, pass_spec_pipeline))

    return InputApp(file_contents, pipeline).run()


if __name__ == "__main__":
    main()
