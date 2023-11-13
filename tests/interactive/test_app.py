from typing import cast

import pytest

from xdsl.backend.riscv.lowering import convert_func_to_riscv_func
from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func, riscv, riscv_func
from xdsl.dialects.builtin import (
    IndexType,
    IntegerAttr,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.interactive.app import InputApp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import ParseError


@pytest.mark.asyncio()
async def test_inputs():
    """Test different inputs produce desired result."""
    async with InputApp().run_test() as pilot:
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
        app = cast(InputApp, pilot.app)

        # Test clicking the "clear passes" button
        app.input_text_area.insert(
            """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """
        )

        # Select a pass
        app.passes_selection_list.select(
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass
        )

        # assert that pass selection affected Output Text Area
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
        # press "clear passes" button
        await pilot.click("#clear_selection_list_button")

        # assder that the Output Text Area and current_module have the expected results
        await pilot.pause()
        assert app.passes_selection_list.selected == []
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

        # Test clicking the "clear input" button
        await pilot.click("#clear_input_button")

        # assert that the input text area has been cleared
        await pilot.pause()
        assert app.input_text_area.text == ""


@pytest.mark.asyncio()
async def test_passes():
    """Test pass application has the desired result."""
    async with InputApp().run_test() as pilot:
        app = cast(InputApp, pilot.app)

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

        # Await on test update to make sure we only update due to pass change later
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

        # Select a pass
        app.passes_selection_list.select(
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass
        )

        # assert that the Output Text Area has changed accordingly
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

        index = IndexType()
        expected_module = ModuleOp(Region([Block()]))
        with ImplicitBuilder(expected_module.body):
            section = riscv.AssemblySectionOp(".text")
            with ImplicitBuilder(section.data):
                riscv.DirectiveOp(".globl", "hello")
                riscv.DirectiveOp(".p2align", "2")
                function = riscv_func.FuncOp(
                    "hello",
                    Region([Block(arg_types=[riscv.Registers.A0])]),
                    ((riscv.Registers.A0,), (riscv.Registers.A0,)),
                )
                with ImplicitBuilder(function.body) as (n,):
                    zero = riscv.MVOp(n, rd=riscv.IntRegisterType(""))
                    n_one = UnrealizedConversionCastOp.get([zero.rd], [index])
                    two = arith.Constant(IntegerAttr(2, index)).result
                    res = arith.Muli(n_one, two)
                    one = UnrealizedConversionCastOp.get(
                        [res.result], [riscv.IntRegisterType("")]
                    )
                    two_two = riscv.MVOp(one, rd=riscv.Registers.A0)
                    riscv_func.ReturnOp(two_two)

        assert isinstance(app.current_module, ModuleOp)
        # Assert that the current module has been changed accordingly
        assert app.current_module.is_structurally_equivalent(expected_module)
