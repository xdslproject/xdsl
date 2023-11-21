from typing import cast

import pytest

from xdsl.backend.riscv.lowering import (
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
)
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
from xdsl.transforms import (
    mlir_opt,
    printf_to_llvm,
)
from xdsl.transforms.experimental import (
    hls_convert_stencil_to_ll_mlir,
)
from xdsl.transforms.experimental.dmp import stencil_global_to_local
from xdsl.utils.exceptions import ParseError


@pytest.mark.asyncio()
async def test_inputs():
    """Test different inputs produce desired result."""
    async with InputApp().run_test() as pilot:
        app = cast(InputApp, pilot.app)

        # clear preloaded code and unselect preselected pass
        app.input_text_area.clear()
        await pilot.pause()

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

        # clear preloaded code and unselect preselected pass
        app.input_text_area.clear()

        await pilot.pause()
        app.input_text_area.insert(
            """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """
        )

        # assert that the Input and Output Text Area's have changed
        await pilot.pause()
        assert (
            app.input_text_area.text
            == """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """
        )
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

        # Test clicking the "clear input" button
        await pilot.click("#clear_input_button")

        # assert that the input text area has been cleared
        await pilot.pause()
        assert app.input_text_area.text == ""

        app.input_text_area.insert(
            """
        func.func @hello(%n : index) -> index {
          %two = arith.constant 2 : index
          %res = arith.muli %n, %two : index
          func.return %res : index
        }
        """
        )

        # Select two passes
        app.pass_pipeline = tuple(
            (
                *app.pass_pipeline,
                convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass,
                convert_arith_to_riscv.ConvertArithToRiscvPass,
            )
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
      %two = riscv.li 2 : () -> !riscv.reg<>
      %two_1 = builtin.unrealized_conversion_cast %two : !riscv.reg<> to index
      %res = builtin.unrealized_conversion_cast %n_1 : index to !riscv.reg<>
      %res_1 = builtin.unrealized_conversion_cast %two_1 : index to !riscv.reg<>
      %res_2 = riscv.mul %res, %res_1 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
      %res_3 = builtin.unrealized_conversion_cast %res_2 : !riscv.reg<> to index
      %1 = builtin.unrealized_conversion_cast %res_3 : index to !riscv.reg<>
      %2 = riscv.mv %1 : (!riscv.reg<>) -> !riscv.reg<a0>
      riscv_func.return %2 : !riscv.reg<a0>
    }
  }
}
"""
        )

        current_pipeline = app.pass_pipeline
        # press "Remove Last Pass" button
        await pilot.click("#remove_last_pass_button")
        await pilot.pause()
        assert app.pass_pipeline == current_pipeline[:-1]

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

        # press "Clear Passes" button
        await pilot.click("#clear_passes_button")

        # assert that the Output Text Area and current_module have the expected results
        await pilot.pause()
        assert app.pass_pipeline == ()
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

        # assert initial state of condense_mode is False
        assert app.condense_mode is False

        # press "Condense" button
        await pilot.click("#condense_button")

        condensed_list = tuple(
            (
                convert_arith_to_riscv.ConvertArithToRiscvPass,
                convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass,
                stencil_global_to_local.DistributeStencilPass,
                hls_convert_stencil_to_ll_mlir.HLSConvertStencilToLLMLIRPass,
                mlir_opt.MLIROptPass,
                printf_to_llvm.PrintfToLLVM,
            )
        )

        await pilot.pause()
        # assert after "Condense Button" is clicked that the state and condensed_pass list change accordingly
        assert app.condense_mode is True
        assert app.available_pass_list == condensed_list

        # press "Uncondense" button
        await pilot.click("#uncondense_button")

        await pilot.pause()
        # assert after "Condense Button" is clicked that the state changes accordingly
        assert app.condense_mode is False


@pytest.mark.asyncio()
async def test_passes():
    """Test pass application has the desired result."""
    async with InputApp().run_test() as pilot:
        app = cast(InputApp, pilot.app)
        # clear preloaded code and unselect preselected pass
        app.input_text_area.clear()

        await pilot.pause()
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
        app.pass_pipeline = tuple(
            (*app.pass_pipeline, convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass)
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
