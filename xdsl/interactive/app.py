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
    Tree,
)
from textual.widgets.tree import TreeNode

from xdsl.dialects import get_all_dialects
from xdsl.dialects.builtin import ModuleOp
from xdsl.interactive.add_arguments_screen import AddArguments
from xdsl.interactive.get_all_available_passes import get_available_pass_list
from xdsl.interactive.load_file_screen import LoadFile
from xdsl.interactive.pass_list_item import PassListItem
from xdsl.interactive.pass_metrics import (
    count_number_of_operations,
    get_diff_operation_count,
)
from xdsl.interactive.passes import (
    AvailablePass,
    apply_passes_to_module,
    get_new_registered_context,
)
from xdsl.ir import Dialect
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.transforms import get_all_passes, individual_rewrite
from xdsl.utils.parse_pipeline import parse_pipeline

from ._pasteboard import pyclip_copy


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
        func.func @hello(%n : i32) -> i32 {
          %two = arith.constant 0 : i32
          %res = arith.addi %two, %n : i32
          func.return %res : i32
        }
        """

    all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...]
    """A dictionary of (uninstantiated) dialects."""

    all_passes: tuple[tuple[str, type[ModulePass]], ...]
    """A dictionary of xDSL passes."""

    current_module = reactive[ModuleOp | Exception | None](None)
    """
    Reactive variable used to save the current state of the modified Input TextArea
    (i.e. is the Output TextArea).
    """
    pass_pipeline = reactive(tuple[ModulePass, ...])
    """Reactive variable that saves the list of selected passes."""

    condense_mode = reactive(False, always_update=True)
    """Reactive boolean."""
    available_pass_list = reactive(tuple[AvailablePass, ...])
    """
    Reactive variable that saves the list of passes that have an effect on
    current_module.
    """

    input_text_area: TextArea
    """Input TextArea."""
    output_text_area: OutputTextArea
    """Output TextArea."""
    selected_passes_list_view: ListView
    """"ListView displaying the selected passes."""
    passes_tree: Tree[type[ModulePass] | ModulePass]
    """Tree displaying the passes available to apply."""

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
    current_file_path: str
    pre_loaded_pass_pipeline: tuple[ModulePass, ...]

    def __init__(
        self,
        all_dialects: tuple[tuple[str, Callable[[], Dialect]], ...],
        all_passes: tuple[tuple[str, type[ModulePass]], ...],
        file_path: str | None = None,
        input_text: str | None = None,
        pass_pipeline: tuple[ModulePass, ...] = (),
    ):
        self.all_dialects = all_dialects
        self.all_passes = all_passes

        if file_path is None:
            self.current_file_path = ""
        else:
            self.current_file_path = file_path

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
        self.input_text_area = TextArea(id="input")
        self.output_text_area = OutputTextArea(id="output")
        self.selected_passes_list_view = ListView(id="selected_passes_list_view")
        self.passes_tree = Tree(label=".", id="passes_tree")
        self.passes_tree.auto_expand = False
        self.input_operation_count_datatable = DataTable(
            id="input_operation_count_datatable"
        )
        self.diff_operation_count_datatable = DataTable(
            id="diff_operation_count_datatable"
        )

        with Horizontal(id="top_container"):
            with Vertical(id="veritcal_tree_selected_passes_list_view"):
                yield self.selected_passes_list_view
                yield self.passes_tree
            with ScrollableContainer(id="buttons"):
                yield Button("Copy Query", id="copy_query_button")
                yield Button("Clear Passes", id="clear_passes_button")
                yield Button("Condense", id="condense_button")
                yield Button("Uncondense", id="uncondense_button")
                yield Button("Remove Last Pass", id="remove_last_pass_button")
                yield Button("Show Operation Count", id="show_operation_count_button")
                yield Button(
                    "Remove Operation Count", id="remove_operation_count_button"
                )
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

        # initialize Tree to contain the pass options
        for n, module_pass in self.all_passes:
            self.passes_tree.root.add(
                label=n,
                data=module_pass,
            )

        # initialize GUI with either specified input text or default example
        self.input_text_area.load_text(self.pre_loaded_input_text)

        # initialize DataTable with column names
        self.input_operation_count_datatable.add_columns("Operation", "Count")
        self.input_operation_count_datatable.zebra_stripes = True

        self.diff_operation_count_datatable.add_columns("Operation", "Count", "Diff")
        self.diff_operation_count_datatable.zebra_stripes = True

        # initialize GUI with specified pass pipeline
        self.pass_pipeline = self.pre_loaded_pass_pipeline

    def compute_available_pass_list(self) -> tuple[AvailablePass, ...]:
        """
        When any reactive variable is modified, this function (re-)computes the
        available_pass_list variable.
        """
        match self.current_module:
            case None:
                return tuple(AvailablePass(p.name, p) for _, p in self.all_passes)
            case Exception():
                return ()
            case ModuleOp():
                return get_available_pass_list(
                    self.all_dialects,
                    self.all_passes,
                    self.input_text_area.text,
                    self.pass_pipeline,
                    self.condense_mode,
                    individual_rewrite.INDIVIDUAL_REWRITE_PATTERNS_BY_NAME,
                )

    def watch_available_pass_list(
        self,
        old_pass_list: tuple[AvailablePass, ...],
        new_pass_list: tuple[AvailablePass, ...],
    ) -> None:
        """
        Function called when the reactive variable available_pass_list changes - updates
        the ListView to display the latest pass options.
        """
        if old_pass_list != new_pass_list:
            self.passes_tree.clear()
            self.expand_node(self.passes_tree.root, new_pass_list)

    def get_root_to_child_pass_list(
        self, expanded_node: TreeNode[type[ModulePass] | ModulePass]
    ) -> tuple[ModulePass, ...]:
        """
        Helper function that returns a pass_pipeline consisiting of the list of nodes
        from the root of the tree, not including the expanded_node child.
        """
        assert expanded_node.data is not None

        pass_list_items: list[type[ModulePass] | ModulePass] = []

        current = expanded_node.parent

        # traverse the path starting from the child node until we reach the root
        while current is not None and current.data is not None and not current.is_root:
            pass_list_items.append(current.data)
            current = current.parent

        root_to_child_pass_list = tuple(
            (
                selected_pass
                if isinstance(selected_pass, ModulePass)
                else selected_pass()
            )
            for selected_pass in reversed(pass_list_items)
        )

        return root_to_child_pass_list

    def update_selected_passes_list_view(self) -> None:
        """
        Helper function that updates the selected passes ListView to display the passes in pass_pipeline.
        """
        self.selected_passes_list_view.clear()
        if len(self.pass_pipeline) >= 1:
            self.selected_passes_list_view.append(ListItem(Label("."), name="."))

        if not self.pass_pipeline:
            return

        # last element is the node of the tree
        pass_pipeline = self.pass_pipeline[:-1]
        for p in pass_pipeline:
            self.selected_passes_list_view.append(
                PassListItem(
                    Label(str(p.pipeline_pass_spec())),
                    module_pass=p,
                    name=p.name,
                )
            )

    def expand_node(
        self,
        expanded_pass: TreeNode[type[ModulePass] | ModulePass],
        child_pass_list: tuple[AvailablePass, ...],
    ) -> None:
        """
        Helper function that adds a subtree to a node, i.e. adds a sub-tree containing
        the child_pass_list with expanded_pass as the root.
        """
        # remove potential children nodes in case expand node has been clicked multiple times on the same node
        expanded_pass.remove_children()

        for pass_name, value in child_pass_list:
            expanded_pass.add(
                label=pass_name,
                data=value,
            )

    def update_root_of_passes_tree(self) -> None:
        """
        Helper function that updates the passes_tree by first resetting the root (to be
        either the "." root if the pass_pipeline is empty or to the last selected pass) and
        updates the subtree of the root.
        """
        # reset rootnode  of tree
        if not self.pass_pipeline:
            self.passes_tree.reset(".")
        else:
            p = self.pass_pipeline[-1]
            self.passes_tree.reset(
                label=str(p.pipeline_pass_spec()),
                data=p,
            )
        # expand the node
        self.expand_node(self.passes_tree.root, self.available_pass_list)

    def get_pass_arguments(
        self,
        selected_pass_value: type[ModulePass],
        root_to_child_pass_list: tuple[ModulePass, ...],
    ) -> None:
        """
        This function facilitates user input of pass concatenated_arg_val by navigating
        to the AddArguments screen, and subsequently parses the returned string upon
        screen dismissal and appends the pass to the pass_pipeline variable.
        """

        def on_exit(
            result: ModulePass | None,
        ) -> None:
            """
            Called when AddArguments Screen is dismissed. This function attempts to
            parse the returned string, and if successful, adds it to the pass_pipeline
            variable.
            """
            if result is not None:
                self.pass_pipeline = (
                    *self.pass_pipeline,
                    *root_to_child_pass_list,
                    result,
                )

        # generates a string containing the concatenated_arg_val and types of the
        # selected pass and initializes the AddArguments Screen to contain the string
        self.push_screen(
            AddArguments(selected_pass_value),
            on_exit,
        )

    @on(Tree.NodeSelected, "#passes_tree")
    def update_pass_pipeline(
        self, event: Tree.NodeSelected[type[ModulePass] | ModulePass]
    ) -> None:
        """
        When a new selection is made, the reactive variable storing the list of selected
        passes is updated.
        """
        selected_pass_node = event.node
        if selected_pass_node.data is None:
            return

        # block ability to select root (i.e. add root pass to the pipeline)
        if selected_pass_node.is_root:
            return

        # get instance
        selected_pass = selected_pass_node.data

        # get root to child passes due to tree traversal possibility
        root_to_child_pass_list = self.get_root_to_child_pass_list(selected_pass_node)

        # if selected_pass_value has arguments, call get_arguments_function to push screen for user input
        if not isinstance(selected_pass, ModulePass) and fields(selected_pass):
            self.get_pass_arguments(selected_pass, root_to_child_pass_list)
        else:
            # if selected_pass_value contains no arguments add the selected pass to pass_pipeline
            if not isinstance(selected_pass, ModulePass):
                selected_pass = selected_pass()
            # selected_pass_value is an "individual_rewrite", add the selected pass to pass_pipeline
            self.pass_pipeline = (
                *self.pass_pipeline,
                *root_to_child_pass_list,
                selected_pass,
            )

    @on(Tree.NodeExpanded, "#passes_tree")
    def expand_tree_node(
        self, event: Tree.NodeExpanded[type[ModulePass] | ModulePass]
    ) -> None:
        """
        Function called when a user expands a node (i.e. a pass) and adds another level
        to the pass selection tree. Allow's multi-level tree traversal.
        """
        expanded_node = event.node
        if expanded_node.data is None:
            self.expand_node(expanded_node, self.available_pass_list)
            return

        # get instance
        selected_pass = expanded_node.data

        if not isinstance(selected_pass, ModulePass):
            if fields(selected_pass):
                # if expanded_pass requires arguments, do not allow node expansion
                return
            else:
                # Get the default/empty pass spec
                selected_pass = selected_pass()

        # if selected_pass_value requires no arguments add the selected pass to pass_pipeline
        root_to_child_pass_list = self.get_root_to_child_pass_list(expanded_node)

        child_pass_pipeline = (
            *self.pass_pipeline,
            *root_to_child_pass_list,
            selected_pass,
        )

        child_pass_list = get_available_pass_list(
            self.all_dialects,
            self.all_passes,
            self.input_text_area.text,
            child_pass_pipeline,
            self.condense_mode,
            individual_rewrite.INDIVIDUAL_REWRITE_PATTERNS_BY_NAME,
        )

        self.expand_node(expanded_node, child_pass_list)

    def watch_pass_pipeline(self) -> None:
        """
        Function called when the reactive variable pass_pipeline changes - updates the
        label to display the respective generated query in the Label.
        """
        self.update_selected_passes_list_view()
        self.update_root_of_passes_tree()
        self.update_current_module()

    @on(TextArea.Changed, "#input")
    def update_current_module(self) -> None:
        """
        Function to parse the input and to apply the list of selected passes to it.
        """
        input_text = self.input_text_area.text
        if (input_text) == "":
            self.current_module = None
            self.current_get_condensed_list = ()
            self.update_input_operation_count_tuple(ModuleOp([], None))
            return
        try:
            ctx = get_new_registered_context(self.all_dialects)
            parser = Parser(ctx, input_text)
            module = parser.parse_module()
            self.update_input_operation_count_tuple(module)
            self.current_module = apply_passes_to_module(
                module, ctx, self.pass_pipeline
            )
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
                Printer(output_stream).print_string(str(e))
                output_text = output_stream.getvalue()
            case ModuleOp():
                output_stream = StringIO()
                printer = Printer(output_stream)
                printer.print_op(self.current_module)
                printer.print_string("\n")

                output_text = output_stream.getvalue()

        self.output_text_area.load_text(output_text)
        self.update_operation_count_diff_tuple()

    def get_query_string(self) -> str:
        """
        Function returning a string containing the textual description of the pass
        pipeline generated thus far.
        """
        if self.current_file_path == "":
            query = "-p "
        else:
            query = self.current_file_path + " -p "

        if self.pass_pipeline:
            query += "'"
            query += ",".join(str(p.pipeline_pass_spec()) for p in self.pass_pipeline)
            query += "'"
        return f"xdsl-opt {query}"

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

    available_passes = ",".join([name for name in get_all_passes()])
    arg_parser.add_argument(
        "-p",
        "--passes",
        required=False,
        help=f"Delimited list of passes. Available passes are: {available_passes}",
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
    pipeline = tuple(PipelinePass.iter_passes(pass_list, pass_spec_pipeline))

    return InputApp(
        tuple(get_all_dialects().items()),
        tuple((p_name, p()) for p_name, p in sorted(get_all_passes().items())),
        file_path,
        file_contents,
        pipeline,
    ).run()


if __name__ == "__main__":
    main()
