from textual import events
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, TextArea


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
        yield TextArea(self.text, id="input")
        yield TextArea(self.text, id="output")
        yield Static("Tree", id="sidebar")

    def on_mount(self) -> None:
        self.query_one("#input", TextArea).border_title = "Input"
        self.query_one("#output", TextArea).border_title = "Output"
        # label.border_title = "Textual Rocks"
        # label.border_subtitle = "Textual Rocks"

    def on_key(self, event: events.Key) -> None:
        self.query_one(RichLog).write(event)

    def on_text_area_changed(self, event: TextArea.Changed):
        self.log(event.text_area.text)


if __name__ == "__main__":
    path = "bla.mlir"
    with open(path) as f:
        text = f.read()
    app = InputApp(text)
    app.run()
