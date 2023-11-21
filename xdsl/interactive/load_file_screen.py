from textual import on
from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
)


class LoadFile(Screen[str]):
    CSS_PATH = "load_file_screen.tcss"

    directory_tree: DirectoryTree

    def __init__(self):
        self.directory_tree = DirectoryTree("./", id="directory_tree")
        super().__init__()

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="directory_tree_container"):
            yield self.directory_tree
            yield Button("Cancel", id="quit_load_file_screen_button")

    def on_mount(self) -> None:
        """Configure widgets in this application before it is first shown."""
        self.query_one("#directory_tree_container").border_title = "Click File to Open"

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        selected_path = str(event.path)
        self.dismiss(selected_path)

    @on(Button.Pressed, "#quit_load_file_screen_button")
    def exit_screen(self, event: Button.Pressed) -> None:
        self.dismiss()
