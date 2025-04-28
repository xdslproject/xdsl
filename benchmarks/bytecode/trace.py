import abc
import copy
import dis
import gc
import inspect
import math
import statistics
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

FUNCTION_DELIMITER_LENGTH = 45
MIN_TIME_PADDING = 100
TRACE_REPEATS = 1000

PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution


@dataclass(repr=False, kw_only=True)
class EventTrace(abc.ABC):
    """Representation of an event in a trace."""

    elapsed: float = field(default=0.0, compare=False)

    def padded_time(self, message: str, prefix: str, elapsed_offset: float) -> str:
        """Get a padded string containing the time for a trace event."""
        resolution_ns = PERF_COUNTER_RESOLUTION * 1000000000
        elapsed_ns = (self.elapsed - elapsed_offset) * 1000000000

        def round_to_resolution(measurement: float, resolution: float) -> float:
            """Round a floating point value to a resolution."""
            if resolution == 0:
                return measurement

            factor = 10 ** math.floor(math.log10(resolution))
            return round(measurement / factor) * factor

        return (
            " " * max(1, MIN_TIME_PADDING - len(message))
            + prefix
            + f" {round_to_resolution(elapsed_ns, resolution_ns):<4} ns"
        )

    @classmethod
    def get_median(cls, traces: list[Self]) -> Self:
        """Get a trace with the median +/- standard error elapsed time."""
        assert len(traces) > 0
        median_trace = copy.copy(traces[0])
        median_trace.elapsed = statistics.median([trace.elapsed for trace in traces])
        return median_trace

    @classmethod
    @abc.abstractmethod
    def from_frame(cls, frame: types.FrameType) -> Self: ...

    @abc.abstractmethod
    def show(
        self, indent: int = 0, show_times: bool = True, elapsed_offset: float = 0
    ) -> str: ...


@dataclass
class OpcodeTrace(EventTrace):
    """Representation of an opcode in a trace."""

    lineno: int | None
    curr_instr: bool
    jump: bool
    offset: int
    opname: str
    arg: int | None
    argrepr: str

    @classmethod
    def get_opcode(cls, frame: types.FrameType) -> dis.Instruction | None:
        """Get the current opcode name for a frame.

        This can be found by retrieving the instruction at the correct offset in
        the frame.
        """
        for instr in dis.get_instructions(frame.f_code):
            if instr.offset == frame.f_lasti:
                return instr
        return None

    @classmethod
    def from_frame(cls, frame: types.FrameType) -> Self:
        """Construct the opcode representation from a frame object."""
        instruction = cls.get_opcode(frame)
        assert instruction is not None

        line_no = getattr(instruction.positions, "lineno", None)
        if line_no is None and instruction.starts_line is not None:
            line_no = instruction.starts_line

        return cls(
            lineno=line_no,
            curr_instr=False,
            jump=instruction.is_jump_target,
            offset=instruction.offset,
            opname=instruction.opname,
            arg=instruction.arg,
            argrepr=instruction.argrepr,
        )

    def show(
        self, indent: int = 0, show_times: bool = True, elapsed_offset: float = 0.0
    ) -> str:
        """Get a string representation of the opcode."""
        curr_instr = "-->" if self.curr_instr else ""
        jump = ">> " if self.jump else ""
        line_no = self.lineno if self.lineno is not None else ""
        arg = self.arg if self.arg is not None else ""
        line = (
            " " * indent
            + f"{line_no:>3} {curr_instr:>3} {jump:>3} {self.offset:<3} "
            + f"{self.opname:<20} {arg:<3} {'(' + self.argrepr + ')':<20} "
        )
        if show_times:
            line += self.padded_time(line, "//", elapsed_offset)
        return line


@dataclass
class LineTrace(EventTrace):
    """Representation of a line in a trace."""

    contents: str

    @classmethod
    def from_frame(cls, frame: types.FrameType) -> Self:
        """Construct the line representation from a frame object."""
        try:
            frameinfo = inspect.getframeinfo(frame)
            code_context = frameinfo.code_context
            assert code_context is not None
            return cls(code_context[0].strip())
        except Exception:
            return cls("<source not available>")

    def show(
        self, indent: int = 0, show_times: bool = True, elapsed_offset: float = 0.0
    ) -> str:
        """Get a string representation of the line."""
        line = f"{' ' * indent}// >>> {self.contents}"
        if show_times:
            line += self.padded_time(line, "##", elapsed_offset)
        return line


@dataclass
class FunctionTrace(EventTrace):
    """Representation of a function in a trace."""

    name: str
    file: Path
    lineno: int | None
    return_time: float | None = None

    @classmethod
    def from_frame(cls, frame: types.FrameType) -> Self:
        """Construct the function representation from a frame object."""
        return cls(
            name=frame.f_code.co_qualname,
            file=Path(frame.f_code.co_filename),
            lineno=frame.f_code.co_firstlineno,
        )

    def show(
        self, indent: int = 0, show_times: bool = True, elapsed_offset: float = 0.0
    ) -> str:
        """Get a string representation of the function."""
        contents = f" {self.file.stem}:{self.lineno} `{self.name}` "
        line = f"\n{' ' * indent}// {contents:{'='}{'^'}{FUNCTION_DELIMITER_LENGTH}}"
        if show_times:
            line += self.padded_time(line, " ==", elapsed_offset)
        return line


@dataclass
class ReturnTrace(EventTrace):
    """Representation of returning from a function in a trace."""

    @classmethod
    def from_frame(cls, frame: types.FrameType) -> Self:
        """Construct the line representation from a frame object."""
        return cls()

    def show(
        self, indent: int = 0, show_times: bool = True, elapsed_offset: float = 0
    ) -> str:
        """Get a string representation of returning from a function."""
        return f"{' ' * (indent)}// {'=' * FUNCTION_DELIMITER_LENGTH}\n"


@dataclass
class BytecodeProfiler:
    _traces: list[EventTrace] = field(default_factory=list)
    _prev_trace_start: float = field(default_factory=time.perf_counter)
    _prev_opcode: OpcodeTrace | None = None
    _prev_line_stack: list[LineTrace] = field(default_factory=list)
    _function_stack: list[FunctionTrace] = field(default_factory=list)
    _calibration_offset: float | None = 0.0

    def reset(self) -> None:
        """Reset the profiler."""
        self.__init__()

    @property
    def _prev_line(self) -> LineTrace | None:
        """Get the previous line being processed.

        A stack is needed to handle lines which invoke new functions, as the
        `POP_TOP` bytecode operation needs to be associated with a line in its
        outer scope.
        """
        if len(self._prev_line_stack) == 0:
            return None
        return self._prev_line_stack[-1]

    @_prev_line.setter
    def _prev_line(self, value: LineTrace) -> None:
        """Add a new line trace to the stack."""
        self._prev_line_stack.append(value)

    def trace_enable_opcode_events(
        self, frame: types.FrameType, _event: str, _arg: Any
    ) -> Callable[..., Any] | None:
        """Enable opcode events on all frames of a function execution trace."""
        frame.f_trace_opcodes = True
        return self.trace_enable_opcode_events

    def trace_all_events(
        self, frame: types.FrameType, event: str, arg: Any
    ) -> Callable[..., Any] | None:
        """Trace all event types on a function."""
        prev_trace_finish = time.perf_counter()
        event_elapsed = prev_trace_finish - self._prev_trace_start
        assert event_elapsed > 0
        frame.f_trace_opcodes = True

        trace: EventTrace | None = None
        if event == "call":
            # Create the new function object and add it to the call stack
            trace = FunctionTrace.from_frame(frame)
            self._function_stack.append(trace)
            # All events contribute to function runtime
            for function in self._function_stack:
                function.elapsed += event_elapsed

        if event == "opcode":
            # Create the new opcode object
            trace = OpcodeTrace.from_frame(frame)
            self._prev_opcode = trace
            # Opcode events contribute to opcode, line, and function runtimes
            self._prev_opcode.elapsed += event_elapsed
            if self._prev_line is not None:
                self._prev_line.elapsed += event_elapsed
            for function in self._function_stack:
                function.elapsed += event_elapsed

        if event == "line":
            # Create the new line object
            trace = LineTrace.from_frame(frame)
            self._prev_line = trace
            # Line events contribute to line, and function runtimes
            self._prev_line.elapsed += event_elapsed  # pyright: ignore[reportOptionalMemberAccess]
            for function in self._function_stack:
                function.elapsed += event_elapsed

        if event == "return":
            # Create the new function object
            trace = ReturnTrace()
            # Return events contribute to return and function runtimes
            trace.elapsed += event_elapsed
            for function in self._function_stack:
                function.elapsed += event_elapsed
            # Pop the returning function from the call stack
            assert len(self._function_stack) > 0
            self._function_stack.pop()
            self._prev_line_stack.pop()
            self._prev_opcode = None

        assert trace is not None
        self._traces.append(trace)

        self._prev_trace_start = time.perf_counter()

        return self.trace_all_events

    def trace_function(
        self, func: Callable[..., Any], *args: list[Any], **kwargs: dict[str, Any]
    ) -> list[EventTrace]:
        """Get the trace of a function."""
        gcold = gc.isenabled()
        gc.disable()
        old_trace = sys.gettrace()

        try:
            # Run once to set `frame.f_trace_opcodes` to generate opcode events
            sys.settrace(self.trace_enable_opcode_events)
            func(*args, **kwargs)

            # Then run a second time to process all the events, including opcodes
            sys.settrace(self.trace_all_events)
            self._prev_trace_start = time.perf_counter()
            func(*args, **kwargs)
            prev_trace_finish = time.perf_counter()

            # Clean up elapsed time for any remaining trace events
            if self._prev_opcode is not None:
                self._prev_opcode.elapsed += prev_trace_finish - self._prev_trace_start
            for remaining_lines in self._prev_line_stack:
                remaining_lines.elapsed += prev_trace_finish - self._prev_trace_start
            for remaining_functions in self._function_stack:
                remaining_functions.elapsed += (
                    prev_trace_finish - self._prev_trace_start
                )
        finally:
            sys.settrace(old_trace)
            if gcold:
                gc.enable()
            profile_traces = [copy.copy(trace) for trace in self._traces]
            self.reset()

        return profile_traces

    def median_trace_function(
        self, func: Callable[..., Any], *args: list[Any], **kwargs: dict[str, Any]
    ) -> list[EventTrace]:
        """Get the median event time across many traces."""
        all_profile_traces: list[list[EventTrace]] = [
            self.trace_function(func, *args, **kwargs) for _ in range(TRACE_REPEATS)
        ]
        return [
            EventTrace.get_median(cast(list[EventTrace], lockstep_traces))
            for lockstep_traces in zip(*all_profile_traces)
        ]

    @classmethod
    def profile(
        cls,
        func: Callable[[], Any],
        indent_size: int = 4,
        show_times: bool = True,
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        """Print a string representation of the profile."""
        profiler = cls()

        calibration_offset = 0
        profile_traces = profiler.median_trace_function(func, *args, **kwargs)

        print(f"//// Trace of `{func.__qualname__}` :")
        indent = -indent_size
        for trace in profile_traces:
            if isinstance(trace, FunctionTrace):
                indent += indent_size
            print(
                trace.show(
                    indent=indent,
                    elapsed_offset=calibration_offset,
                    show_times=show_times,
                )
            )
            if isinstance(trace, ReturnTrace):  # TODO: Also for exceptions
                indent -= indent_size


def print_bytecode(
    func: Callable[[], Any], indent_size: int = 4, show_times: bool = True
):
    """Print the bytecode executed when running a function."""
    BytecodeProfiler.profile(func, indent_size=indent_size)


if __name__ == "__main__":
    from xdsl.irdl import (
        IRDLOperation,
        irdl_op_definition,
        traits_def,
    )
    from xdsl.traits import OpTrait

    class TraitA(OpTrait):
        """An example trait."""

    @irdl_op_definition
    class HasTraitAOp(IRDLOperation):
        """An operation which has a trait A."""

        name = "has_trait_a"
        traits = traits_def(TraitA())

    HAS_TRAIT_A_OP = HasTraitAOp()

    def has_trait_function():
        HAS_TRAIT_A_OP.has_trait(TraitA)

    def inner_function(x: int | str | float):
        assert x

    def example_function():
        inner_function(1)
        inner_function("Hello")
        inner_function(5.0)

    def go():
        example_function()

    print_bytecode(go)
