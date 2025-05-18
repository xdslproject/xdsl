import abc
import dis
import gc
import inspect
import math
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FUNC_WARMUPS = 2000
FUNCTION_DELIMITER_LENGTH = 45
MIN_TIME_PADDING = 100

PYTHON_VERSION = sys.version_info
PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution


@dataclass(eq=True, frozen=True)
class StopEvent:
    """An event with a previous timestamp."""

    prev_timestamp: float
    now_timestamp: float
    exception: Exception | None


@dataclass(eq=True, frozen=True)
class _LogMessage:
    """An event with a previous timestamp."""

    message: Any


@dataclass(eq=True, frozen=True)
class TracedEvent(abc.ABC):
    """An event in a trace, having both a current and previous timestamp."""

    prev_timestamp: float
    now_timestamp: float

    @classmethod
    @abc.abstractmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: Any,
        prev_timestamp: float,
        now_timestamp: float,
    ) -> "TracedEvent": ...


@dataclass(eq=True, frozen=True)
class CallEvent(TracedEvent):
    """A call event in a trace."""

    name: str
    file: Path
    lineno: int | None

    @classmethod
    def get_name(cls, frame: types.FrameType) -> str:
        """Get the name of a function."""
        if PYTHON_VERSION.minor <= 10:
            return frame.f_code.co_name
        return frame.f_code.co_qualname

    @classmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: None,
        prev_timestamp: float,
        now_timestamp: float,
    ) -> "CallEvent":
        """Construct the representation from a frame object."""
        return cls(
            prev_timestamp=prev_timestamp,
            now_timestamp=now_timestamp,
            name=cls.get_name(frame),
            file=Path(frame.f_code.co_filename),
            lineno=frame.f_code.co_firstlineno,
        )


@dataclass(eq=True, frozen=True)
class LineEvent(TracedEvent):
    """A call event in a trace."""

    contents: str

    @classmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: None,
        prev_timestamp: float,
        now_timestamp: float,
    ) -> "LineEvent":
        """Construct the representation from a frame object."""
        try:
            frameinfo = inspect.getframeinfo(frame)
            code_context = frameinfo.code_context
            assert code_context is not None
            contents = code_context[0].strip()
        except Exception:
            contents = "<source not available>"

        return cls(
            prev_timestamp=prev_timestamp,
            now_timestamp=now_timestamp,
            contents=contents,
        )


@dataclass(eq=True, frozen=True)
class ReturnEvent(TracedEvent):
    """A call event in a trace."""

    return_value: Any

    @classmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: None,
        prev_timestamp: float,
        now_timestamp: float,
    ) -> "ReturnEvent":
        """Construct the representation from a frame object."""
        return cls(
            prev_timestamp=prev_timestamp, now_timestamp=now_timestamp, return_value=arg
        )


@dataclass(eq=True, frozen=True)
class ExceptionEvent(TracedEvent):
    """A call event in a trace."""

    exception: type
    value: Exception = field(compare=False)
    traceback: types.TracebackType | None = field(compare=False)

    @classmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: tuple[type, Exception, types.TracebackType | None],
        prev_timestamp: float,
        now_timestamp: float,
    ) -> "ExceptionEvent":
        """Construct the representation from a frame object."""
        return cls(
            prev_timestamp=prev_timestamp,
            now_timestamp=now_timestamp,
            exception=arg[0],
            value=arg[1],
            traceback=arg[2],
        )


@dataclass(eq=True, frozen=True)
class OpcodeEvent(TracedEvent):
    """A call event in a trace."""

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
    def get_lineno(cls, instruction: dis.Instruction) -> int | None:
        """Get the line number for an instruction."""
        if PYTHON_VERSION.minor <= 10:
            return instruction.starts_line
        if instruction.line_number is not None:
            return instruction.line_number
        return getattr(instruction.positions, "lineno", None)

    @classmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: None,
        prev_timestamp: float,
        now_timestamp: float,
    ) -> "OpcodeEvent":
        """Construct the representation from a frame object."""
        instruction = cls.get_opcode(frame)
        assert instruction is not None

        return cls(
            prev_timestamp=prev_timestamp,
            now_timestamp=now_timestamp,
            lineno=cls.get_lineno(instruction),
            curr_instr=False,
            jump=instruction.is_jump_target,
            offset=instruction.offset,
            opname=instruction.opname,
            arg=instruction.arg,
            argrepr=instruction.argrepr,
        )


EVENT_NAME_LOOKUP: dict[str, type[TracedEvent]] = {
    "call": CallEvent,
    "line": LineEvent,
    "return": ReturnEvent,
    "exception": ExceptionEvent,
    "opcode": OpcodeEvent,
}


def round_to_resolution(measurement: float, resolution: float) -> float:
    """Round a floating point value to a resolution."""
    if resolution == 0:
        return measurement

    factor = 10 ** math.floor(math.log10(resolution))
    return round(measurement / factor) * factor


def padded_time(message: str, prefix: str, elapsed: float) -> str:
    """Get a padded string containing the time for a trace event."""
    resolution_ns = PERF_COUNTER_RESOLUTION * 1000000000
    elapsed_ns = elapsed * 1000000000

    return (
        " " * max(1, MIN_TIME_PADDING - len(message))
        + prefix
        + f" {round_to_resolution(elapsed_ns, resolution_ns):<4} ns"
    )


@dataclass
class BytecodeProfiler:
    """A profiler for Python (>=3.7) bytecode."""

    events: list[TracedEvent | StopEvent | _LogMessage] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    _prev_timestamp: float | None = None
    uninstrumented_time: float | None = None
    empty_instrumented_time: float | None = None
    instrumented_time: float | None = None

    def reset(self) -> None:
        """Reset the profiler."""
        self.__init__()

    def _trace__empty(
        self, frame: types.FrameType, _event: str, _arg: Any
    ) -> Callable[..., Any] | None:
        """Trace function which does nothing."""
        return self._trace__empty

    def _trace__collect_all_events(
        self, frame: types.FrameType, event: str, arg: Any
    ) -> Callable[..., Any] | None:
        """Collect all events emitted by the function being traced."""
        now_timestamp = time.perf_counter()
        frame.f_trace_opcodes = True

        assert self._prev_timestamp is not None
        assert event in EVENT_NAME_LOOKUP
        self.events.append(
            EVENT_NAME_LOOKUP[event].from_frame(
                frame=frame,
                arg=arg,
                prev_timestamp=self._prev_timestamp,
                now_timestamp=now_timestamp,
            )
        )

        self._prev_timestamp = time.perf_counter()
        return self._trace__collect_all_events

    def profile(
        self, func: Callable[..., Any], *args: list[Any], **kwargs: dict[str, Any]
    ) -> None:
        """Collect all events emitted when invoking a function."""
        if PYTHON_VERSION.minor < 10:
            raise RuntimeError("Tracing only supported for Python versions >=3.10!")

        gcold = gc.isenabled()
        gc.disable()
        old_trace = sys.gettrace()
        exception = None

        try:
            # Warmup cache/specialising adaptive interpreter/JIT
            for _ in range(FUNC_WARMUPS):
                try:
                    func(*args, **kwargs)
                except Exception as _exc:
                    pass

            # Measure the time taken to run the function once without tracing
            start_timestamp: float | None = None
            try:
                start_timestamp = time.perf_counter()
                func(*args, **kwargs)
            except Exception as _exc:
                pass
            finally:
                finish_timestamp = time.perf_counter()
                self.uninstrumented_time = (
                    finish_timestamp - start_timestamp
                    if start_timestamp is not None
                    else 0.0
                )

            # Measure the time taken to run the function once with empty tracing
            start_timestamp = None
            try:
                sys.settrace(self._trace__empty)
                start_timestamp = time.perf_counter()
                func(*args, **kwargs)
            except Exception as _exc:
                pass
            finally:
                finish_timestamp = time.perf_counter()
                self.empty_instrumented_time = (
                    finish_timestamp - start_timestamp
                    if start_timestamp is not None
                    else 0.0
                )
                sys.settrace(old_trace)

            # Measure the time taken to run the function once with tracing
            try:
                sys.settrace(self._trace__collect_all_events)
                self._prev_timestamp = time.perf_counter()
                func(*args, **kwargs)
            except Exception as exc:
                exception = exc
            finally:
                finish_timestamp = time.perf_counter()

        finally:
            sys.settrace(old_trace)
            if gcold:
                gc.enable()

        self.events.append(
            StopEvent(
                prev_timestamp=(
                    self._prev_timestamp
                    if self._prev_timestamp is not None
                    else finish_timestamp
                ),
                now_timestamp=finish_timestamp,
                exception=exception,
            )
        )

        self.instrumented_time = (
            (finish_timestamp - self.events[0].now_timestamp)
            if len(self.events) and isinstance(self.events[0], TracedEvent)
            else 0.0
        )

    @classmethod
    def calculate_elapsed_times(
        cls,
        events: list[TracedEvent | StopEvent | _LogMessage],
        uninstrumented_time: float | None,
        empty_instrumented_time: float | None,
        instrumented_time: float | None,
    ) -> dict[TracedEvent | StopEvent | _LogMessage, float]:
        """Construct timing information from the event sequence."""
        assert uninstrumented_time is not None
        assert empty_instrumented_time is not None
        assert instrumented_time is not None

        elapsed_times: dict[TracedEvent | StopEvent | _LogMessage, float] = {}

        tracing_time = 0.0
        for event in events:
            elapsed_times[event] = 0.0
            if isinstance(event, TracedEvent) or isinstance(event, StopEvent):
                tracing_time += event.now_timestamp - event.prev_timestamp

        prev_opcode: OpcodeEvent | None = None
        for event in events:
            if isinstance(event, OpcodeEvent):
                if prev_opcode is not None:
                    elapsed_times[prev_opcode] += (
                        event.now_timestamp - event.prev_timestamp
                    )
                prev_opcode = event
            elif isinstance(event, StopEvent):
                if prev_opcode is not None:
                    elapsed_times[prev_opcode] += (
                        event.now_timestamp - event.prev_timestamp
                    )

        print(f"Uninstrumented = {uninstrumented_time * 1000000000:.1f}ns")
        print(f"Empty instrumented = {empty_instrumented_time * 1000000000:.1f}ns")
        print(f"Instrumented = {instrumented_time * 1000000000:.1f}ns")
        print(f"Tracing = {tracing_time * 1000000000:.1f}ns")
        print(
            f"Uninstrumented + Tracing = {(uninstrumented_time + tracing_time) * 1000000000:.1f}ns"
        )
        print(f"Non-tracing = {(instrumented_time - tracing_time) * 1000000000:.1f}ns")
        # TODO: Good idea: trace event creation and times separately with some
        # good key to recombine them (or switch ot namedtuples/cython?)
        average_nontracing_time = (instrumented_time - tracing_time) / len(events)

        for event in events:
            if isinstance(event, OpcodeEvent):
                print(
                    f"Measured = {(elapsed_times[event] - average_nontracing_time) * 1000000000:.1f}ns,\t"
                    f"Original = {elapsed_times[event] * 1000000000:.1f}ns,\t"
                    f"Offset = {average_nontracing_time * 1000000000:.1f}ns,\t"
                )

        return elapsed_times

    @classmethod
    def print_events(
        cls,
        events: list[TracedEvent | StopEvent | _LogMessage],
        elapsed_times: dict[TracedEvent | StopEvent | _LogMessage, float],
        trace_name: str,
        indent_size: int = 4,
    ) -> None:
        """Print a string representation of the traced events."""

        # Print the events with this timing information
        print(f"//// Trace of `{trace_name}` :")
        for event in events:
            if isinstance(event, OpcodeEvent):
                curr_instr = "-->" if event.curr_instr else ""
                jump = ">> " if event.jump else ""
                line_no = event.lineno if event.lineno is not None else ""
                arg = event.arg if event.arg is not None else ""
                line = (
                    f"{line_no:>3} {curr_instr:>3} {jump:>3} {event.offset:<3} "
                    f"{event.opname:<20} {arg:<3} {'(' + event.argrepr + ')':<20} "
                )
                print(line + padded_time(line, "//", elapsed_times[event]))


def print_bytecode(
    func: Callable[[], Any],
    *args: list[Any],
    **kwargs: dict[str, Any],
) -> None:
    """Print the bytecode executed when running a function."""
    profiler = BytecodeProfiler()
    profiler.profile(func, *args, **kwargs)

    _elapsed_times = BytecodeProfiler.calculate_elapsed_times(
        events=profiler.events,
        uninstrumented_time=profiler.uninstrumented_time,
        empty_instrumented_time=profiler.empty_instrumented_time,
        instrumented_time=profiler.instrumented_time,
    )
    # BytecodeProfiler.print_events(
    #     profiler.events, elapsed_times, trace_name=func.__qualname__
    # )


if __name__ == "__main__":

    def inner_function(x: int | str | float):
        assert x

    def raise_exception():
        raise ValueError("Help")

    def example_function():
        inner_function(1)
        inner_function("Hello")
        raise_exception()
        inner_function(5.0)

    def go():
        example_function()

    print_bytecode(go)


### Old implementation ###
#
# def _calculate_elapsed_times(
#     self,
# ) -> dict[TracedEvent | StopEvent | _LogMessage, float]:
#     """Construct timing information from the event sequence."""
#     elapsed_times: dict[TracedEvent | StopEvent | _LogMessage, float] = {}

#     prev_line_stack: list[LineEvent] = []
#     call_stack: list[CallEvent] = []
#     for event in self._events:
#         elapsed_times[event] = 0.0

#     for event in self._events:
#         if isinstance(event, OpcodeEvent):
#             elapsed_times[event] += event.now_timestamp - event.prev_timestamp
#     return elapsed_times

#     for event in self._events:
#         if isinstance(event, StopEvent) or isinstance(event, _LogMessage):
#             continue

#         event_elapsed = event.now_timestamp - event.prev_timestamp
#         if isinstance(event, CallEvent):
#             # Function events contribute to function runtime
#             call_stack.append(event)
#             for function in call_stack:
#                 elapsed_times[function] += event_elapsed
#         elif isinstance(event, LineEvent):
#             # Line events contribute to line and function runtimes
#             prev_line_stack.append(event)
#             elapsed_times[event] += event_elapsed
#             for function in call_stack:
#                 elapsed_times[function] += event_elapsed
#         elif isinstance(event, ReturnEvent):
#             # Return events contribute to return and function runtimes
#             elapsed_times[event] += event_elapsed
#             for function in call_stack:
#                 elapsed_times[function] += event_elapsed
#             # Pop the returning function from the call stack
#             assert len(call_stack) > 0
#             call_stack.pop()
#             # Exiting a function starts affecting the invoking line/opcode again
#             if len(prev_line_stack) > 0:
#                 prev_line_stack.pop()
#                 # TODO: prev opcode?
#         elif isinstance(event, ExceptionEvent):
#             # Exception events contribute to exception and function runtimes
#             elapsed_times[event] += event_elapsed
#             for function in call_stack:
#                 elapsed_times[function] += event_elapsed
#             # TODO: prev line and opcode?
#         elif isinstance(event, OpcodeEvent):
#             # Opcode events contribute to opcode, line, and function runtimes
#             elapsed_times[event] += event_elapsed
#             if len(prev_line_stack) > 0:
#                 elapsed_times[prev_line_stack[-1]] += event_elapsed
#             for function in call_stack:
#                 elapsed_times[function] += event_elapsed
#     return elapsed_times

# def _print_events(self, trace_name: str, indent_size: int = 4) -> None:
#     """Print a string representation of the traced events."""

#     def padded_time(message: str, prefix: str, elapsed: float) -> str:
#         """Get a padded string containing the time for a trace event."""
#         resolution_ns = PERF_COUNTER_RESOLUTION * 1000000000
#         elapsed_ns = elapsed * 1000000000

#         def round_to_resolution(measurement: float, resolution: float) -> float:
#             """Round a floating point value to a resolution."""
#             if resolution == 0:
#                 return measurement

#             factor = 10 ** math.floor(math.log10(resolution))
#             return round(measurement / factor) * factor

#         return (
#             " " * max(1, MIN_TIME_PADDING - len(message))
#             + prefix
#             + f" {round_to_resolution(elapsed_ns, resolution_ns):<4} ns"
#         )

#     # Get the timing information
#     elapsed_times = self._calculate_elapsed_times()

#     # Print the events with this timing information
#     print(f"//// Trace of `{trace_name}` :")
#     for event in self._events:
#         if isinstance(event, OpcodeEvent):
#             curr_instr = "-->" if event.curr_instr else ""
#             jump = ">> " if event.jump else ""
#             line_no = event.lineno if event.lineno is not None else ""
#             arg = event.arg if event.arg is not None else ""
#             line = (
#                 f"{line_no:>3} {curr_instr:>3} {jump:>3} {event.offset:<3} "
#                 f"{event.opname:<20} {arg:<3} {'(' + event.argrepr + ')':<20} "
#             )
#             print(line + padded_time(line, "//", elapsed_times[event]))
#     return None

#     indent = -indent_size
#     for event in self._events:
#         if isinstance(event, CallEvent):
#             indent += indent_size
#             contents = f" {event.file.stem}:{event.lineno} `{event.name}` "
#             line = f"\n{' ' * indent}// =={contents:{'='}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}=="
#             print(line + " " + padded_time(line, "==", elapsed_times[event]))
#         elif isinstance(event, LineEvent):
#             line = f"{' ' * indent}// >>> {event.contents}"
#             print(line + padded_time(line, "##", elapsed_times[event]))
#         elif isinstance(event, ReturnEvent):
#             print(f"{' ' * (indent)}// {'=' * FUNCTION_DELIMITER_LENGTH}\n")
#             indent -= indent_size
#         elif isinstance(event, ExceptionEvent):
#             contents = f" `{event.exception.__qualname__}` "
#             line = f"\n{' ' * indent}// !!{contents:{'!'}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}!!"
#             print(line + " " + padded_time(line, "==", elapsed_times[event]))
#         elif isinstance(event, OpcodeEvent):
#             curr_instr = "-->" if event.curr_instr else ""
#             jump = ">> " if event.jump else ""
#             line_no = event.lineno if event.lineno is not None else ""
#             arg = event.arg if event.arg is not None else ""
#             line = (
#                 " " * indent
#                 + f"{line_no:>3} {curr_instr:>3} {jump:>3} {event.offset:<3} "
#                 + f"{event.opname:<20} {arg:<3} {'(' + event.argrepr + ')':<20} "
#             )
#             print(line + padded_time(line, "//", elapsed_times[event]))
#         else:
#             pass  # TODO: Handle stop and log messages
