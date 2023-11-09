from typing import cast

import pytest
from textual.pilot import Pilot

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.interactive.app import InputApp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import ParseError


@pytest.mark.asyncio()
async def test_input():
    """Test pressing keys has the desired result."""
    async with InputApp().run_test() as pilot:
        pilot: Pilot[None] = pilot
        app = cast(InputApp, pilot.app)

        # Test no input
        assert app.output_text_area.text == "No input"
        assert app.current_module is None

        # Test inccorect input
        app.input_text_area.insert("dkjfd")
        await pilot.pause()
        assert (
            app.output_text_area.text
            == "(Span[5:6](text=''), 'Operation builtin.unregistered does not have a custom format.')"
        )
        assert isinstance(app.current_module, ParseError)
        assert (
            str(app.current_module)
            == "(Span[5:6](text=''), 'Operation builtin.unregistered does not have a custom format.')"
        )

        # Test corect input
        app.input_text_area.clear()
        app.input_text_area.insert(
            """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """
        )
        await pilot.pause()
        assert (
            app.output_text_area.text
            == """builtin.module {
  func.func @hello(%n : index) -> index {
    %two = arith.constant 2 : index
    %res = arith.muli %n, %two : index
    func.return %res : index
  }
}
"""
        )

        index = IndexType()

        expected_module = ModuleOp(Region([Block()]))
        with ImplicitBuilder(expected_module.body):
            function = func.FuncOp("hello", ((index,), (index,)))
            with ImplicitBuilder(function.body) as (n,):
                two = arith.Constant(IntegerAttr(2, index)).result
                res = arith.Muli(n, two)
                func.Return(res)

        assert isinstance(app.current_module, ModuleOp)
        assert app.current_module.is_structurally_equivalent(expected_module)


@pytest.mark.asyncio()
async def test_buttons():
    """Test pressing keys has the desired result."""
    async with InputApp().run_test() as pilot:
        pilot: Pilot[None] = pilot
        app = cast(InputApp, pilot.app)

        # Test clicking the "clear input" button
        app.input_text_area.insert(
            """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """
        )
        await pilot.click("#clear_input_button")
        # assert that the curent_module and test_module's are structurally equivalent
        await pilot.pause()
        assert app.input_text_area.text == ""
