from typing import cast

import pytest
from textual.pilot import Pilot

from xdsl.backend.riscv.lowering import convert_func_to_riscv_func
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.interactive.app import InputApp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import ParseError


@pytest.mark.asyncio()
async def tests_interactive():
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

        await pilot.pause()
        assert isinstance(app.current_module, ModuleOp)
        assert app.current_module.is_structurally_equivalent(expected_module)


@pytest.mark.asyncio()
async def test_buttons_and_passes():
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
        # press clear input button
        await pilot.click("#clear_input_button")

        # assert that the curent_module and test_module's are structurally equivalent
        await pilot.pause()
        assert app.input_text_area.text == ""

        # Testing a pass
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
        assert app.input_text_area != ""

        # Select a pass
        app.passes_selection_list.select(
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass
        )

        await pilot.pause()
        assert app.output_text_area != app.input_text_area
        assert str(app.current_module) != "No input"
        await pilot.pause()
        assert (
            app.output_text_area.text
            == """builtin.module {
  riscv.assembly_section ".text" {
    riscv.directive ".globl" "hello"
    riscv.directive ".p2align" "2"
    riscv_func.func @hello(%n : !riscv.reg<a0>) -> !riscv.reg<a0> {
      %0 = riscv.mv %n : (!riscv.reg<a0>) -> !riscv.reg<>
      %n_1 = builtin.unrealized_conversion_cast %0 : !riscv.reg<> to index
      %two = arith.constant 2 : index
      %res = arith.muli %n_1, %two : index
      %1 = builtin.unrealized_conversion_cast %res : index to !riscv.reg<>
      %2 = riscv.mv %1 : (!riscv.reg<>) -> !riscv.reg<a0>
      riscv_func.return %2 : !riscv.reg<a0>
    }
  }
}
"""
        )
