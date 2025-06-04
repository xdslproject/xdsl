from dataclasses import dataclass
from typing import Any, cast

import pytest
from textual.app import App

from xdsl.interactive.add_arguments_screen import AddArguments
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class TestPass(ModulePass):
    """Test pass with required arguments for testing AddArguments screen"""

    name = "testpass"

    required_arg: int
    optional_arg: str | None = None

    def apply(self, ctx: Any, op: Any) -> None:
        pass


class LaunchScreenApp(App[None]):
    """A minimal app that can launch the AddArguments screen for testing."""


@pytest.mark.asyncio
async def test_enter_button_enabled_state():
    """Test that enter button is enabled/disabled based on argument validity"""
    async with LaunchScreenApp().run_test() as pilot:
        app = cast(LaunchScreenApp, pilot.app)

        screen = AddArguments(TestPass)

        app.push_screen(screen)
        await pilot.pause()

        # Initially disabled with empty args
        assert screen.argument_text_area.text == "required_arg=int optional_arg=none"
        assert screen.selected_pass_value is None
        assert screen.enter_button.disabled

        # Invalid args - missing required field
        screen.argument_text_area.load_text('optional_arg="test"')
        await pilot.pause()
        assert screen.enter_button.disabled

        # Invalid args - wrong type
        screen.argument_text_area.load_text('required_arg="not_a_number"')
        await pilot.pause()
        assert screen.enter_button.disabled

        # Valid args - only required
        screen.argument_text_area.load_text("required_arg=42")
        await pilot.pause()
        assert not screen.enter_button.disabled

        # Valid args - with optional
        screen.argument_text_area.load_text('required_arg=42 optional_arg="hello"')
        await pilot.pause()
        assert screen.selected_pass_value == TestPass(42, "hello")

        assert not screen.enter_button.disabled

        # Invalid args - syntax error
        screen.argument_text_area.load_text("required_arg=42,")
        await pilot.pause()
        assert screen.enter_button.disabled
