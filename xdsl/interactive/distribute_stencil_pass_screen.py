from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.widgets import Button, SelectionList, TextArea

from xdsl.transforms.experimental.dmp import stencil_global_to_local
from xdsl.transforms.experimental.dmp.decompositions import (
    DomainDecompositionStrategy,
)


class DistributeStencilPassScreen(Screen[dict[str, list[int]]]):
    CSS_PATH = "distribute_stencil_pass_screen.tcss"

    stencil_argument_text_area: TextArea
    strategy_argument_selection_list: SelectionList[str]

    STRATEGIES: dict[
        str, type[DomainDecompositionStrategy]
    ] = stencil_global_to_local.DistributeStencilPass.STRATEGIES

    res: dict[str, list[int]]

    def __init__(self):
        self.stencil_argument_text_area = TextArea("", id="stencil_argument_text_area")
        self.strategy_argument_selection_list = SelectionList(
            id="strategy_argument_selection_list"
        )
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="container"):
            with Horizontal(id="text_area_horizontal"):
                yield self.stencil_argument_text_area
                yield self.strategy_argument_selection_list
            with Horizontal(id="cancel_enter_buttons"):
                yield Button("Clear Text", id="clear_input_button")
                yield Button("Enter", id="enter_button")
                yield Button("Cancel", id="quit_screen_button")

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        self.query_one(
            "#stencil_argument_text_area"
        ).border_title = "Provide Number of Slices."
        self.query_one(
            "#strategy_argument_selection_list"
        ).border_title = "Select one decomposition strategy."
        self.query_one(
            "#container"
        ).border_title = (
            "distribute-stencil requires two arguments 'Slices`' and 'Strategy'."
        )

        self.stencil_argument_text_area.load_text(
            "Please provide number of slices to decompose the input into as a list of valid integers separated by spaces."
        )

        selections = sorted((name, name) for (name, _) in self.STRATEGIES.items())
        self.strategy_argument_selection_list.add_options(selections)

    @on(Button.Pressed, "#quit_screen_button")
    def exit_screen(self, event: Button.Pressed) -> None:
        self.dismiss()

    @on(Button.Pressed, "#clear_input_button")
    def clear_text_area(self, event: Button.Pressed) -> None:
        self.stencil_argument_text_area.clear()

    @on(Button.Pressed, "#enter_button")
    def enter_arguments(self, event: Button.Pressed) -> None:
        slices_input_string = self.stencil_argument_text_area.text

        # Get correct slices argument
        try:
            # Splitting the string using spaces as the delimiter and converting each element to an integer
            slices_res = [int(num) for num in slices_input_string.split()]

            # if only one selection has been made
            if len(self.strategy_argument_selection_list.selected) == 1:
                selected_strategy = self.strategy_argument_selection_list.selected[0]
                self.res = {selected_strategy: slices_res}
                self.dismiss(self.res)

            # if no selection has been made or if more than one selection has been made
            else:
                self.strategy_argument_selection_list.deselect_all
                self.stencil_argument_text_area.load_text(
                    "Error: Please select one strategy."
                )
                return
        except ValueError as e:
            self.stencil_argument_text_area.load_text(f"Error: {e}")
            self.stencil_argument_text_area.load_text(
                "Ensure that the input string contains valid integers separated by spaces."
            )
            return
