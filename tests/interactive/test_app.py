from io import StringIO
from typing import cast

import pytest
from textual.pilot import Pilot

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, builtin, func
from xdsl.dialects.builtin import IndexType, IntegerAttr, ModuleOp
from xdsl.interactive.app import InputApp
from xdsl.ir import Block, MLContext, Region
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PipelinePass
from xdsl.printer import Printer
from xdsl.tools.command_line_tool import get_all_dialects
from xdsl.utils.exceptions import ParseError


def transform_input_return_module(
    input_text: str, passes: list[type[ModulePass]]
) -> builtin.ModuleOp | Exception:
    """
    Function called when the Input TextArea is changed or a pass is selected/unselected. This function parses the Input IR, applies selected passes and updates
    the current_module reactive variable.
    """
    try:
        ctx = MLContext(True)
        for dialect in get_all_dialects():
            ctx.load_dialect(dialect)
        parser = Parser(ctx, input_text)
        module = parser.parse_module()

        pipeline = PipelinePass([p() for p in passes])
        pipeline.apply(ctx, module)

        res = module
    except Exception as e:
        res = e

    return res


def transform_input_return_str(res: ModuleOp | Exception) -> str:
    match res:
        case None:
            output_text = "No input"
        case Exception() as e:
            output_stream = StringIO()
            Printer(output_stream).print(e)
            output_text = output_stream.getvalue()
        case ModuleOp():
            output_stream = StringIO()
            Printer(output_stream).print(res)
            output_text = output_stream.getvalue()

    return output_text


@pytest.mark.asyncio()
async def test_interactive():
    """Test interactive app has the desired results."""
    async with InputApp().run_test() as pilot:
        pilot: Pilot[None] = pilot
        app = cast(InputApp, pilot.app)

        """Test input/output without passes."""
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
        if isinstance(app.current_module, ModuleOp):
            assert app.current_module.is_structurally_equivalent(expected_module)

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
        assert str(
            app.current_module
            == "(Span[0:1](text=''), 'Could not parse entire input!')"
        )
        assert (
            app.output_text_area.text
            == "(Span[0:1](text=''), 'Could not parse entire input!')"
        )

        pass_options = [
            app.passes_selection_list.get_option_at_index(i)
            for i in range(app.passes_selection_list.option_count)
        ]

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

            # test pass selection works
            app.passes_selection_list.select(selection)

            test_module = transform_input_return_module(
                app.input_text_area.text, app.passes_selection_list.selected
            )
            test_output_area_str = transform_input_return_str(test_module)

            await pilot.pause()
            # assert (str(test_module) == str(app.current_module))
            assert test_output_area_str == app.output_text_area.text

            # assert that the curent_module and test_module's are structurally equivalent
            assert isinstance(app.current_module and test_module, ModuleOp | Exception)
            if isinstance(test_module, ModuleOp):
                assert app.current_module.is_structurally_equivalent(test_module)

            # test "Clear Passes" button works
            selected_pass = app.passes_selection_list.select(selection)

            await pilot.click("#clear_selection_list_button")

            test_module = transform_input_return_module(
                app.input_text_area.text, app.passes_selection_list.selected
            )
            test_output_area_str = transform_input_return_str(test_module)

            # assert that the clear button clears the selection list
            await pilot.pause()
            assert app.passes_selection_list.selected is not selected_pass

            # assert that the input has returned to its original state
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

            # assert that the curent_module and test_module's are structurally equivalent
            await pilot.pause()
            assert isinstance(app.current_module and test_module, ModuleOp)
            if isinstance(test_module, ModuleOp):
                assert app.current_module.is_structurally_equivalent(test_module)

                # assert that the input and output text area's are equal
            await pilot.pause()
            assert app.output_text_area.text == test_output_area_str


# @pytest.mark.asyncio()
# async def test_input():
#     """Test pressing keys has the desired result."""
#     async with InputApp().run_test() as pilot:
#         pilot: Pilot[None] = pilot
#         app = cast(InputApp, pilot.app)
#         # Test no input
#         assert app.output_text_area.text == "No input"
#         assert app.current_module is None

#         # Test inccorect input
#         app.input_text_area.insert("dkjfd")
#         await pilot.pause()
#         assert (
#             app.output_text_area.text
#             == "(Span[5:6](text=''), 'Operation builtin.unregistered does not have a custom format.')"
#         )
#         assert isinstance(app.current_module, ParseError)
#         assert (
#             str(app.current_module)
#             == "(Span[5:6](text=''), 'Operation builtin.unregistered does not have a custom format.')"
#         )

#         # Test correct input
#         app.input_text_area.clear()
#         app.input_text_area.insert(
#             """
#         func.func @hello(%n : index) -> index {
#           %two = arith.constant 2 : index
#           %res = arith.muli %n, %two : index
#           func.return %res : index
#         }
#         """
#         )
#         await pilot.pause()
#         assert (
#             app.output_text_area.text
#             == """builtin.module {
#   func.func @hello(%n : index) -> index {
#     %two = arith.constant 2 : index
#     %res = arith.muli %n, %two : index
#     func.return %res : index
#   }
# }
# """
#         )

#         index = IndexType()

#         expected_module = ModuleOp(Region([Block()]))
#         with ImplicitBuilder(expected_module.body):
#             function = func.FuncOp("hello", ((index,), (index,)))
#             with ImplicitBuilder(function.body) as (n,):
#                 two = arith.Constant(IntegerAttr(2, index)).result
#                 res = arith.Muli(n, two)
#                 func.Return(res)

#         await pilot.pause()
#         if isinstance(app.current_module, ModuleOp):
#             assert app.current_module.is_structurally_equivalent(
#                 expected_module)

# @pytest.mark.asyncio()
# async def test_passes():
#     """Test selected passes."""
#     async with InputApp().run_test() as pilot:
#         pilot: Pilot[None] = pilot
#         app = cast(InputApp, pilot.app)

#         pass_options = [
#             app.passes_selection_list.get_option_at_index(i)
#             for i in range(app.passes_selection_list.option_count)
#         ]

#         for selection in pass_options:

#             app.input_text_area.clear()
#             app.input_text_area.insert(
#                 """
#             func.func @hello(%n : index) -> index {
#             %two = arith.constant 2 : index
#             %res = arith.muli %n, %two : index
#             func.return %res : index
#             }
#             """
#             )

#             # test pass selection
#             app.passes_selection_list.select(selection)

#             test_module = transform_input_return_module(
#                 app.input_text_area.text, app.passes_selection_list.selected)
#             test_output_area_str = transform_input_return_str(test_module)

#             await pilot.pause()
#             # assert (str(test_module) == str(app.current_module))
#             assert (test_output_area_str == app.output_text_area.text)

#             # assert that the curent_module and test_module's are structurally equivalent
#             assert isinstance(
#                 app.current_module and test_module, ModuleOp | Exception)
#             if isinstance(app.current_module, ModuleOp) and isinstance(
#                 test_module, ModuleOp
#             ):
#                 assert app.current_module.is_structurally_equivalent(
#                     test_module)


# @pytest.mark.asyncio()
# async def test_buttons():
#     """Test pressing keys has the desired result."""
#     async with InputApp().run_test() as pilot:
#         pilot: Pilot[None] = pilot
#         app = cast(InputApp, pilot.app)


#                 # Test clicking the "clear input" button
#         app.input_text_area.insert(
#             """
#         func.func @hello(%n : index) -> index {
#           %two = arith.constant 2 : index
#           %res = arith.muli %n, %two : index
#           func.return %res : index
#         }
#         """
#         )
#         await pilot.click("#clear_input_button")
#         # assert that the curent_module and test_module's are structurally equivalent
#         await pilot.pause()
#         assert (app.input_text_area.text == "")
#         assert str(app.current_module ==
#                    "(Span[0:1](text=''), 'Could not parse entire input!')")
#         assert (app.output_text_area.text ==
#                 "(Span[0:1](text=''), 'Could not parse entire input!')")

#         # Test clicking the "clear passes" button
#         pass_options = [
#             app.passes_selection_list.get_option_at_index(i)
#             for i in range(app.passes_selection_list.option_count)
#         ]

#         for selection in pass_options:

#             app.input_text_area.clear()
#             app.input_text_area.load_text(
#                 """
#             func.func @hello(%n : index) -> index {
#             %two = arith.constant 2 : index
#             %res = arith.muli %n, %two : index
#             func.return %res : index
#             }
#             """
#             )

#             selected_pass = app.passes_selection_list.select(selection)

#             await pilot.click("#clear_selection_list_button")

#             test_module = transform_input_return_module(
#                 app.input_text_area.text, app.passes_selection_list.selected)
#             test_output_area_str = transform_input_return_str(test_module)

#             # assert that the clear button clears the selection list
#             await pilot.pause()
#             assert app.passes_selection_list.selected is not selected_pass

#             # assert that the input has returned to its original state
#             await pilot.pause()
#             assert app.input_text_area.text == """
#             func.func @hello(%n : index) -> index {
#             %two = arith.constant 2 : index
#             %res = arith.muli %n, %two : index
#             func.return %res : index
#             }
#             """

#             # assert that the curent_module and test_module's are structurally equivalent
#             await pilot.pause()
#             assert isinstance(
#                 app.current_module and test_module, ModuleOp)
#             if isinstance(app.current_module, ModuleOp) and isinstance(
#                 test_module, ModuleOp
#             ):
#                 assert app.current_module.is_structurally_equivalent(
#                     test_module)

#                 # assert that the input and output text area's are equal
#             await pilot.pause()
#             assert app.output_text_area.text == test_output_area_str
