from io import StringIO
from typing import cast

import pytest
from textual.pilot import Pilot
from textual.widgets import TextArea

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.interactive.app import InputApp
from xdsl.ir import Block, MLContext, Region
from xdsl.parser import Parser
from xdsl.passes import PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects
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

        # Test correct input
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
        if isinstance(app.current_module, ModuleOp) and isinstance(
            expected_module, ModuleOp
        ):
            assert app.current_module.is_structurally_equivalent(expected_module)


@pytest.mark.asyncio()
async def test_passes():
    """Test selected passes."""
    async with InputApp().run_test() as pilot:
        pilot: Pilot[None] = pilot
        app = cast(InputApp, pilot.app)

        pass_options = [
            app.passes_selection_list.get_option_at_index(i)
            for i in range(app.passes_selection_list.option_count)
        ]

        # Test that the application of one pass at a time provides the corrent output and current_module
        for selection in pass_options:
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

            # Build identical input text area for testing
            test_input_text_area = TextArea(id="test_input_text_area")
            test_input_text_area.clear()
            test_input_text_area.insert(
                """
            func.func @hello(%n : index) -> index {
            %two = arith.constant 2 : index
            %res = arith.muli %n, %two : index
            func.return %res : index
            }
            """
            )

            app.passes_selection_list.select(selection)
            selected_passes = app.passes_selection_list.selected

            try:
                ctx = MLContext(True)
                for dialect in get_all_dialects():
                    ctx.load_dialect(dialect)
                parser = Parser(ctx, test_input_text_area.text)
                module = parser.parse_module()

                pipeline = PipelinePass([p() for p in selected_passes])
                pipeline.apply(ctx, module)

                test_module = module
            except Exception as e:
                test_module = e

            # build an identical output TextArea for testing
            test_output_text_area = TextArea(id="test_output_text_area")

            match test_module:
                case None:
                    output_text = "No input"
                case Exception() as e:
                    output_stream = StringIO()
                    Printer(output_stream).print(e)
                    output_text = output_stream.getvalue()
                case ModuleOp():
                    output_stream = StringIO()
                    Printer(output_stream).print(test_module)
                    output_text = output_stream.getvalue()
            test_output_text_area.load_text(output_text)

            await pilot.pause()
            # assert that the current_module and output is the expected output (i.e. the generated test output) after a pass is applied
            assert app.input_text_area.text == test_input_text_area.text

            # assert that the curent_module and test_module's are structurally equivalent
            await pilot.pause()
            assert isinstance(app.current_module and test_module, ModuleOp | Exception)
            if isinstance(app.current_module, ModuleOp) and isinstance(
                test_module, ModuleOp
            ):
                assert app.current_module.is_structurally_equivalent(test_module)

            # assert that the output text area and test output text area are the same
            assert app.output_text_area.text == test_output_text_area.text

        # Test that the application of two passes at a time provides the corrent output and current_module
        for selection_one in pass_options:
            for selection_two in pass_options:
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
                # Build identical input text area for testing
                test_input_text_area = TextArea(id="test_input_text_area")
                test_input_text_area.clear()
                test_input_text_area.insert(
                    """
                func.func @hello(%n : index) -> index {
                %two = arith.constant 2 : index
                %res = arith.muli %n, %two : index
                func.return %res : index
                }
                """
                )

                app.passes_selection_list.select(selection_one)
                app.passes_selection_list.select(selection_two)
                selected_passes = app.passes_selection_list.selected

                try:
                    ctx = MLContext(True)
                    for dialect in get_all_dialects():
                        ctx.load_dialect(dialect)
                    parser = Parser(ctx, test_input_text_area.text)
                    module = parser.parse_module()

                    pipeline = PipelinePass([p() for p in selected_passes])
                    pipeline.apply(ctx, module)

                    test_module = module
                except Exception as e:
                    test_module = e

                # build an identical output TextArea for testing
                test_output_text_area = TextArea(id="test_output_text_area")

                match test_module:
                    case None:
                        output_text = "No input"
                    case Exception() as e:
                        output_stream = StringIO()
                        Printer(output_stream).print(e)
                        output_text = output_stream.getvalue()
                    case ModuleOp():
                        output_stream = StringIO()
                        Printer(output_stream).print(test_module)
                        output_text = output_stream.getvalue()
                test_output_text_area.load_text(output_text)

                await pilot.pause()
                # assert that the current_module and output is the expected output (i.e. the generated test output) after two passes are applied
                assert app.input_text_area.text == test_input_text_area.text

                # assert that the curent_module and test_module's are structurally equivalent
                await pilot.pause()
                assert isinstance(
                    app.current_module and test_module, ModuleOp | Exception
                )
                if isinstance(app.current_module, ModuleOp) and isinstance(
                    test_module, ModuleOp
                ):
                    assert app.current_module.is_structurally_equivalent(test_module)

                # assert that the output text area and test output text area are the same
                assert app.output_text_area.text == test_output_text_area.text


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
        # Build identical input text area for testing
        test_input_text_area = TextArea(id="test_input_text_area")

        await pilot.click("#clear_input_button")

        test_input_text_area.clear()
        try:
            ctx = MLContext(True)
            for dialect in get_all_dialects():
                ctx.load_dialect(dialect)
            parser = Parser(ctx, test_input_text_area.text)
            module = parser.parse_module()

            test_module = module
        except Exception as e:
            test_module = e

            # assert that the curent_module and test_module's are structurally equivalent
        await pilot.pause()
        assert isinstance(app.current_module and test_module, ModuleOp | Exception)
        if isinstance(app.current_module, ModuleOp) and isinstance(
            test_module, ModuleOp
        ):
            assert app.current_module.is_structurally_equivalent(test_module)

        pass_options = [
            app.passes_selection_list.get_option_at_index(i)
            for i in range(app.passes_selection_list.option_count)
        ]

        # Test clicking the "clear passes" button
        for selection in pass_options:
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

            test_input_text_area.clear()
            test_input_text_area.insert(
                """
            func.func @hello(%n : index) -> index {
            %two = arith.constant 2 : index
            %res = arith.muli %n, %two : index
            func.return %res : index
            }
            """
            )

            app.passes_selection_list.select(selection)
            selected_passes = app.passes_selection_list.selected

            try:
                ctx = MLContext(True)
                for dialect in get_all_dialects():
                    ctx.load_dialect(dialect)
                parser = Parser(ctx, test_input_text_area.text)
                module = parser.parse_module()

                pipeline = PipelinePass([p() for p in selected_passes])
                pipeline.apply(ctx, module)

                test_module = module
            except Exception as e:
                test_module = e

                # build an identical output TextArea for testing
            test_output_text_area = TextArea(id="test_output_text_area")

            match test_module:
                case None:
                    output_text = "No input"
                case Exception() as e:
                    output_stream = StringIO()
                    Printer(output_stream).print(e)
                    output_text = output_stream.getvalue()
                case ModuleOp():
                    output_stream = StringIO()
                    Printer(output_stream).print(test_module)
                    output_text = output_stream.getvalue()
            test_output_text_area.load_text(output_text)

            await pilot.click("#clear_selection_list_button")

            app.passes_selection_list.deselect_all()
            selected_passes = app.passes_selection_list.selected

            try:
                ctx = MLContext(True)
                for dialect in get_all_dialects():
                    ctx.load_dialect(dialect)
                parser = Parser(ctx, test_input_text_area.text)
                module = parser.parse_module()

                pipeline = PipelinePass([p() for p in selected_passes])
                pipeline.apply(ctx, module)

                test_module = module
            except Exception as e:
                test_module = e

            match test_module:
                case None:
                    output_text = "No input"
                case Exception() as e:
                    output_stream = StringIO()
                    Printer(output_stream).print(e)
                    output_text = output_stream.getvalue()
                case ModuleOp():
                    output_stream = StringIO()
                    Printer(output_stream).print(test_module)
                    output_text = output_stream.getvalue()
            test_output_text_area.load_text(output_text)

            # assert that the clear button clears the selection list
            await pilot.pause()
            assert app.passes_selection_list.selected is not selected_passes

            # assert that the input and test_input are equal
            await pilot.pause()
            assert app.input_text_area.text == test_input_text_area.text

            # assert that the curent_module and test_module's are structurally equivalent
            await pilot.pause()
            assert isinstance(app.current_module and test_module, ModuleOp | Exception)
            if isinstance(app.current_module, ModuleOp) and isinstance(
                test_module, ModuleOp
            ):
                assert app.current_module.is_structurally_equivalent(test_module)

                # assert that the input and output text area's are equal
            await pilot.pause()
            assert app.output_text_area.text == test_output_text_area.text
