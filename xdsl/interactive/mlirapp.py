"""
An interactive command-line tool to explore compilation pipeline construction.

Execute `mlir-gui` in your terminal to run it.

Run `textual run xdsl.interactive.mlirapp:MLIRApp --dev` to run in development mode. Please
be sure to install `textual-dev` to run this command.
"""

import argparse
import os
from collections.abc import Callable
from io import StringIO
from typing import Any, ClassVar

from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Label,
    ListItem,
    ListView,
    TextArea,
    Tree,
)
from textual.widgets.tree import TreeNode

from xdsl.dialects import get_all_dialects
from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.add_arguments_screen import AddArguments
from xdsl.interactive.load_file_screen import LoadFile
from xdsl.interactive.mlir_helper import (
    apply_mlir_pass_with_args_to_module,
    generate_mlir_pass,
    get_mlir_pass_list,
    get_new_registered_context,
)
from xdsl.ir import Dialect
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.utils.exceptions import PassPipelineParseError
from xdsl.utils.parse_pipeline import parse_pipeline

from ._pasteboard import pyclip_copy


class OutputTextArea(TextArea):
    """Used to prevent users from being able to alter the Output TextArea."""

    async def _on_key(self, event: events.Key) -> None:
        event.prevent_default()


class MLIRApp(App[None]):
    """
    Interactive application for constructing compiler pipelines.
    """

    CSS_PATH = "app.tcss"

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit_app", "Quit"),
    ]

    SCREENS: ClassVar[dict[str, type[Screen[Any]] | Callable[[], Screen[Any]]]] = {
        "load_file": LoadFile,
        "add_arguments_screen": AddArguments,
    }
    """
    A dictionary that maps names on to Screen objects.
    """

    INITIAL_IR_TEXT = """builtin.module {
    func.func @hello(%n : i32) -> i32 {
        %b = arith.addi %n, %n : i32
        %c = arith.muli %b, %b : i32
        func.return %b : i32
    }
}
    """

    all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...]
    """A dictionary of (uninstantiated) dialects."""
    all_passes: tuple[str, ...]
    """A dictionary of MLIR passes."""
    pass_pipeline = reactive(tuple[str, ...])
    """Reactive variable that saves the list of selected passes."""

    current_module = reactive[ModuleOp | Exception | None](None)
    """
    Reactive variable used to save the current state of the modified Input TextArea
    (i.e. is the Output TextArea).
    """

    input_text_area: TextArea
    """Input TextArea."""
    output_text_area: OutputTextArea
    """Output TextArea."""
    selected_passes_list_view: ListView
    """"ListView displaying the selected passes."""
    passes_tree: Tree[str]
    """Tree displaying the passes available to apply."""

    pre_loaded_input_text: str
    current_file_path: str
    pre_loaded_pass_pipeline: tuple[str, ...]

    def __init__(
        self,
        all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...],
        all_passes: tuple[str, ...],
        file_path: str | None = None,
        input_text: str | None = None,
        pass_pipeline: tuple[str, ...] = (),
    ):
        self.all_dialects = all_dialects
        self.all_passes = all_passes

        if file_path is None:
            self.current_file_path = ""
        else:
            self.current_file_path = file_path

        if input_text is None:
            self.pre_loaded_input_text = MLIRApp.INITIAL_IR_TEXT
        else:
            self.pre_loaded_input_text = input_text

        self.pre_loaded_pass_pipeline = pass_pipeline

        super().__init__()

    def compose(self) -> ComposeResult:
        """
        Creates the required widgets, events, etc.
        """
        self.input_text_area = TextArea(id="input")
        self.output_text_area = OutputTextArea(id="output")
        self.selected_passes_list_view = ListView(id="selected_passes_list_view")
        self.passes_tree = Tree(label="mlir-opt", id="passes_tree")
        self.passes_tree.expand = False

        with Horizontal(id="top_container"):
            with Vertical(id="veritcal_tree_selected_passes_list_view"):
                yield self.selected_passes_list_view
                yield self.passes_tree
            with ScrollableContainer(id="buttons"):
                yield Button("Copy Query", id="copy_query_button")
                yield Button("Clear Passes", id="clear_passes_button")
                yield Button("Remove Last Pass", id="remove_last_pass_button")
        with Horizontal(id="bottom_container"):
            with Horizontal(id="input_horizontal_container"):
                with Vertical(id="input_container"):
                    yield self.input_text_area
                    with Horizontal(id="input_horizontal"):
                        yield Button("Clear Input", id="clear_input_button")
                        yield Button("Load File", id="load_file_button")

            with Horizontal(id="output_horizontal_container"):
                with Vertical(id="output_container"):
                    yield self.output_text_area
                    yield Button("Copy Output", id="copy_output_button")
        yield Footer()

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        # Registers the theme for the Input/Output TextAreas
        self.input_text_area.theme = "vscode_dark"
        self.output_text_area.theme = "vscode_dark"

        # add titles for various widgets
        self.query_one("#input_container").border_title = "Input MLIR IR"
        self.query_one("#output_container").border_title = "Output MLIR IR"

        # initialize Tree to contain the pass options
        for pass_name in self.all_passes:
            self.passes_tree.root.add(
                label=pass_name,
                data=(pass_name, None),
            )

        # initialize GUI with either specified input text or default example
        self.input_text_area.load_text(self.pre_loaded_input_text)

    def expand_node(
        self,
        expanded_pass: TreeNode[tuple[str, ...]],
        child_pass_list: tuple[str, ...],
    ) -> None:
        """
        Helper function that adds a subtree to a node, i.e. adds a sub-tree containing the child_pass_list with expanded_pass as the root.
        """
        # remove potential children nodes in case expand node has been clicked multiple times on the same node
        expanded_pass.remove_children()

        for pass_name in child_pass_list:
            expanded_pass.add(
                label=pass_name,
                data=(pass_name, None),
            )

    def update_root_of_passes_tree(self) -> None:
        """
        Helper function that updates the passes_tree by first resetting the root (to be
        either the "." root if the pass_pipeline is empty or to the last selected pass) and
        updates the subtree of the root.
        """
        # reset rootnode  of tree
        if self.pass_pipeline == ():
            self.passes_tree.reset(".")
        else:
            value = self.pass_pipeline[-1]
            self.passes_tree.reset(
                label=str(value),
                data=(value, None),
            )
        # expand the node
        self.expand_node(self.passes_tree.root, self.all_passes)

    def update_selected_passes_list_view(self) -> None:
        """
        Helper function that updates the selected passes ListView to display the passes in pass_pipeline.
        """
        self.selected_passes_list_view.clear()
        if len(self.pass_pipeline) >= 1:
            self.selected_passes_list_view.append(ListItem(Label("mlir-opt"), name="."))

        if not self.pass_pipeline:
            return

        # last element is the node of the tree
        pass_pipeline = self.pass_pipeline[:-1]
        for pass_value in pass_pipeline:
            self.selected_passes_list_view.append(
                ListItem(Label(str(pass_value)), name=pass_value)
            )

    @on(TextArea.Changed, "#input")
    def update_current_module(self) -> None:
        """
        Function to parse the input and to apply the list of selected passes to it.
        """
        input_text = self.input_text_area.text
        if (input_text) == "":
            self.current_module = None
            self.current_get_condensed_list = ()
            return
        try:
            ctx = get_new_registered_context(self.all_dialects)
            parser = Parser(ctx, input_text)
            module = parser.parse_module()
            current_mlir_opt_pass = generate_mlir_pass(self.pass_pipeline)
            self.current_module = apply_mlir_pass_with_args_to_module(
                module, ctx, current_mlir_opt_pass
            )
        except Exception as e:
            self.current_module = e

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

    def watch_pass_pipeline(self) -> None:
        """
        Function called when the reactive variable pass_pipeline changes - updates the
        label to display the respective generated query in the Label.
        """
        self.update_selected_passes_list_view()
        self.update_root_of_passes_tree()
        self.update_current_module()

    def get_pass_arguments(
        self,
        selected_pass_value: type[ModulePass],
    ) -> None:
        """
        This function facilitates user input of pass concatenated_arg_val by navigating
        to the AddArguments screen, and subsequently parses the returned string upon
        screen dismissal and appends the pass to the pass_pipeline variable.
        """

        def add_pass_with_arguments_to_pass_pipeline(
            concatenated_arg_val: str | None,
        ) -> None:
            """
            Called when AddArguments Screen is dismissed. This function attempts to parse
            the returned string, and if successful, adds it to the pass_pipeline variable.
            In case of parsing failure, the AddArguments Screen is pushed, revealing the
            Parse Error.
            """
            try:
                # if screen was dismissed and user 1) cleared the screen 2) made no changes
                if concatenated_arg_val is None:
                    return

                self.pass_pipeline = (*self.pass_pipeline, concatenated_arg_val)
                return

            except PassPipelineParseError as e:
                error = f"PassPipelineParseError: {e}"

            screen = AddArguments(TextArea(error, id="argument_text_area"))
            self.push_screen(screen, add_pass_with_arguments_to_pass_pipeline)

        # generates a string containing the concatenated_arg_val and types of the selected pass and initializes the AddArguments Screen to contain the string
        self.push_screen(
            AddArguments(
                TextArea(
                    selected_pass_value,
                    id="argument_text_area",
                )
            ),
            add_pass_with_arguments_to_pass_pipeline,
        )

    @on(Tree.NodeSelected, "#passes_tree")
    def update_pass_pipeline(self, event: Tree.NodeSelected[tuple[str, ...]]) -> None:
        """
        When a new selection is made, the reactive variable storing the list of selected
        passes is updated.
        """
        selected_pass = event.node
        if selected_pass.data is None:
            return

        # block ability to select root (i.e. add root pass to the pipeline)
        if selected_pass.is_root:
            return

        # get instance
        selected_pass_value, selected_pass_spec = selected_pass.data

        if "=" in selected_pass_value:
            # Add pass with arguments to pass pipeline
            self.get_pass_arguments(selected_pass_value)

        else:  # Add pass without arguments to pass pipeline
            self.pass_pipeline = (*self.pass_pipeline, selected_pass_value)

    def get_query_string(self) -> str:
        """
        Function returning a string containing the textual description of the pass
        pipeline generated thus far.
        """
        query = self.current_file_path

        if self.pass_pipeline:
            query += " ".join(val for val in self.pass_pipeline)

        return f"mlir-opt {query}"

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )

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
        self.pass_pipeline = tuple[str, ...]()

    @on(Button.Pressed, "#remove_last_pass_button")
    def remove_last_pass(self, event: Button.Pressed) -> None:
        """Last selected pass removed when "Remove Last Pass" button is pressed."""
        self.pass_pipeline = self.pass_pipeline[:-1]

    @on(Button.Pressed, "#load_file_button")
    def load_file(self, event: Button.Pressed) -> None:
        """
        Pushes screen displaying DirectoryTree widget when "Load File" button is pressed.
        """

        def check_load_file(file_path: str | None) -> None:
            """
            Called when LoadFile is dismissed. Loads selected file into
            input_text_area.
            """
            if file_path is None:
                return

            # Clear Input TextArea and Pass Pipeline
            self.pass_pipeline = ()
            self.input_text_area.clear()

            try:
                if os.path.exists(file_path):
                    # Open the file and read its contents
                    with open(file_path) as file:
                        file_contents = file.read()
                        self.input_text_area.load_text(file_contents)
                    self.current_file_path = file_path
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

    available_passes = ",".join([name for name in get_mlir_pass_list()])
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
    pass_list = get_mlir_pass_list()
    pipeline = tuple(PipelinePass.build_pipeline_tuples(pass_list, pass_spec_pipeline))

    return MLIRApp(
        tuple(get_all_dialects().items()),
        get_mlir_pass_list(),
        file_path,
        file_contents,
        pipeline,
    ).run()


if __name__ == "__main__":
    main()
