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
from time import perf_counter
from typing import Any, TypeAlias

FUNC_WARMUPS = 2000
FUNCTION_DELIMITER_LENGTH = 45
MIN_TIME_PADDING = 100

PYTHON_VERSION = sys.version_info
PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution


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


@dataclass(eq=True, frozen=True)
class StopEvent:
    """An event guarding the end of the trace."""

    exception: Exception | None


@dataclass(eq=True, frozen=True)
class TracedEvent(abc.ABC):
    """An event in a trace, having both a current and previous timestamp."""

    @classmethod
    @abc.abstractmethod
    def from_frame(
        cls,
        frame: types.FrameType,
        arg: Any,
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
    ) -> "CallEvent":
        """Construct the representation from a frame object."""
        return cls(
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
    ) -> "ReturnEvent":
        """Construct the representation from a frame object."""
        return cls(return_value=arg)


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
    ) -> "ExceptionEvent":
        """Construct the representation from a frame object."""
        return cls(
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
    ) -> "OpcodeEvent":
        """Construct the representation from a frame object."""
        instruction = cls.get_opcode(frame)
        assert instruction is not None

        return cls(
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


EventTimestamp: TypeAlias = tuple[str, float | None, float | None]


@dataclass
class BytecodeProfiler:
    """A profiler for Python (>=3.7) bytecode."""

    events: list[TracedEvent | StopEvent] = field(default_factory=list)
    timestamps: list[EventTimestamp] = field(default_factory=list)
    _prev_event_timestamp: float | None = None
    elapsed_times: dict[TracedEvent | StopEvent, float] = field(default_factory=dict)
    uninstrumented_time: float | None = None
    instrumented_time: float | None = None

    def reset(self) -> None:
        """Reset the profiler."""
        self.__init__()

    def _trace__collect_event_timestamps(
        self, frame: types.FrameType, _event: str, _arg: Any
    ) -> Callable[..., Any] | None:
        """Trace function which does nothing."""
        now_timestamp = perf_counter()
        frame.f_trace_opcodes = True

        # TODO: Add something further to make id unique...
        self.timestamps.append(
            (
                _event,
                self._prev_event_timestamp,
                now_timestamp,
            )
        )

        self._prev_event_timestamp = perf_counter()
        return self._trace__collect_event_timestamps

    def _trace__collect_all_events(
        self, frame: types.FrameType, event: str, arg: Any
    ) -> Callable[..., Any] | None:
        """Collect all events emitted by the function being traced."""
        frame.f_trace_opcodes = True

        assert event in EVENT_NAME_LOOKUP
        self.events.append(
            EVENT_NAME_LOOKUP[event].from_frame(
                frame=frame,
                arg=arg,
            )
        )

        return self._trace__collect_all_events

    def profile(
        self, func: Callable[..., Any], *args: list[Any], **kwargs: dict[str, Any]
    ) -> None:
        """Collect all events emitted when invoking a function."""
        if PYTHON_VERSION.minor < 10:
            raise RuntimeError("Tracing only supported for Python versions >=3.10!")

        old_trace = sys.gettrace()
        gcold = gc.isenabled()
        gc.disable()

        exception = None
        finish_timestamp: float | None = None
        try:
            # Measure the time taken to run the function once without tracing
            start_timestamp: float | None = None
            try:
                start_timestamp = perf_counter()
                func(*args, **kwargs)
            except Exception as _exc:
                pass
            finally:
                finish_timestamp = perf_counter()
                self.uninstrumented_time = (
                    finish_timestamp - start_timestamp
                    if start_timestamp is not None
                    else None
                )

            # Trace the function collecting the event data to corrollate
            try:
                sys.settrace(self._trace__collect_all_events)
                func(*args, **kwargs)
            except Exception as exc:
                exception = exc
            finally:
                sys.settrace(None)

            # Trace the function collecting only the timestamps
            start_timestamp = None
            try:
                sys.settrace(self._trace__collect_event_timestamps)
                start_timestamp = perf_counter()
                func(*args, **kwargs)
            except Exception as exc:
                exception = exc
            finally:
                sys.settrace(None)
                finish_timestamp = perf_counter()
                self.instrumented_time = (
                    finish_timestamp - start_timestamp
                    if start_timestamp is not None
                    else None
                )

        finally:
            if gcold:
                gc.enable()
            sys.settrace(old_trace)

        self.events.append(StopEvent(exception))
        self.timestamps.append(("stop", finish_timestamp, None))

        self.elapsed_times = self._calculate_elapsed_times()

    def _calculate_elapsed_times(self) -> dict[TracedEvent | StopEvent, float]:
        """Calculate the elapsed time for each event."""
        assert len(self.events) == len(self.timestamps)

        native_tracing_time = 0.0
        for i in range(len(self.timestamps)):
            if i < len(self.events) - 1:
                assert (start_tracing_timestamp := self.timestamps[i][2]) is not None
                assert (end_tracing_timestamp := self.timestamps[i + 1][1]) is not None
                native_tracing_time += end_tracing_timestamp - start_tracing_timestamp

        print(f"Uninstrumented = {self.uninstrumented_time * 1000000000:.1f}ns")
        print(f"Instrumented = {self.instrumented_time * 1000000000:.1f}ns")
        print(f"Tracing = {native_tracing_time * 1000000000:.1f}ns")
        print(
            f"Uninstrumented + Tracing = {(self.uninstrumented_time + native_tracing_time) * 1000000000:.1f}ns"
        )
        print(
            f"Non-tracing = {(self.instrumented_time - native_tracing_time) * 1000000000:.1f}ns"
        )

        average_nontracing_time = (self.instrumented_time - native_tracing_time) / len(
            self.events
        )
        print(f"Offset = {average_nontracing_time * 1000000000:.1f}ns")

        elapsed_times: dict[TracedEvent | StopEvent, float] = {}
        for i, timestamp in enumerate(self.timestamps[:-1]):
            if i == 0:
                continue
            assert (start_execution_timestamp := timestamp[1]) is not None
            assert (end_execution_timestamp := timestamp[2]) is not None
            elapsed_times[self.events[i - 1]] = (
                end_execution_timestamp - start_execution_timestamp
            )
        elapsed_times[self.events[-1]] = 0

        return elapsed_times

    @classmethod
    def print_events(
        cls,
        events: list[Any],
        elapsed_times: dict[Any, float],
        trace_name: str,
        indent_size: int = 4,
    ) -> None:
        """Print a string representation of the traced events."""

        # Print the events with this timing information
        print(f"//// Trace of `{trace_name}` :")

        indent = -indent_size
        for event in events:
            # print(event, elapsed_times[event])
            if isinstance(event, CallEvent):
                indent += indent_size
                contents = f" {event.file.stem}:{event.lineno} `{event.name}` "
                line = f"\n{' ' * indent}// =={contents:{'='}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}=="
                print(line + " " + padded_time(line, "==", elapsed_times[event]))
            elif isinstance(event, LineEvent):
                line = f"{' ' * indent}// >>> {event.contents}"
                print(line + padded_time(line, "##", elapsed_times[event]))
            elif isinstance(event, ReturnEvent):
                print(f"{' ' * (indent)}// {'=' * FUNCTION_DELIMITER_LENGTH}\n")
                indent -= indent_size
            elif isinstance(event, ExceptionEvent):
                contents = f" `{event.exception.__qualname__}` "
                line = f"\n{' ' * indent}// !!{contents:{'!'}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}!!"
                print(line + " " + padded_time(line, "==", elapsed_times[event]))
            if isinstance(event, OpcodeEvent):
                # TODO: Lift back into dataclasses
                curr_instr = "-->" if event.curr_instr else ""
                jump = ">> " if event.jump else ""
                line_no = event.lineno if event.lineno is not None else ""
                arg = event.arg if event.arg is not None else ""
                line = (
                    " " * indent
                    + f"{line_no:>3} {curr_instr:>3} {jump:>3} {event.offset:<3} "
                    + f"{event.opname:<20} {arg:<3} {'(' + event.argrepr + ')':<20} "
                )
                print(line + padded_time(line, "//", elapsed_times[event]))


def print_bytecode(
    func: Callable[[], Any],
    *args: list[Any],
    **kwargs: dict[str, Any],
) -> None:
    """Print the bytecode executed when running a function.

    The function must be pure, as it is run many times to get various peices of
    data.
    """
    profiler = BytecodeProfiler()

    # Warmup cache/specialising adaptive interpreter/JIT
    for _ in range(FUNC_WARMUPS):
        try:
            func(*args, **kwargs)
        except Exception as _exc:
            pass

    events = None
    all_elapsed_times = []
    for _ in range(3):
        profiler.reset()
        profiler.profile(func, *args, **kwargs)
        if events is not None:
            assert events == profiler.events
        events = profiler.events
        all_elapsed_times.append(profiler.elapsed_times)

    elapsed_times = {}
    for elapsed_time in all_elapsed_times:
        for event, time in elapsed_time.items():
            if event in elapsed_times:
                elapsed_times[event] += time
            else:
                elapsed_times[event] = 0.0
    print(elapsed_times)
    # for event in elapsed_times:
    #     print(event, elapsed_times[event], len(all_elapsed_times))
    #     elapsed_times[event] /= len(all_elapsed_times)

    BytecodeProfiler.print_events(events, elapsed_times, trace_name=func.__qualname__)


if __name__ == "__main__":

    def inner_function(x: int | str | float):
        assert x

    def raise_exception():
        raise ValueError("Help")

    def example_function():
        inner_function(1)
        inner_function("Hello")
        x = perf_counter()
        # raise_exception()
        inner_function(5.0)

    def go():
        example_function()

    print_bytecode(go)


# import abc
# import dis
# import gc
# import inspect
# import math
# import sys
# import time
# import types
# from collections import namedtuple
# from collections.abc import Callable
# from dataclasses import dataclass, field
# from pathlib import Path
# from time import perf_counter
# from typing import Any, TypeAlias

# FUNC_WARMUPS = 2000
# FUNCTION_DELIMITER_LENGTH = 45
# MIN_TIME_PADDING = 100

# PYTHON_VERSION = sys.version_info
# PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution


# def round_to_resolution(measurement: float, resolution: float) -> float:
#     """Round a floating point value to a resolution."""
#     if resolution == 0:
#         return measurement

#     factor = 10 ** math.floor(math.log10(resolution))
#     return round(measurement / factor) * factor


# def padded_time(message: str, prefix: str, elapsed: float) -> str:
#     """Get a padded string containing the time for a trace event."""
#     resolution_ns = PERF_COUNTER_RESOLUTION * 1000000000
#     elapsed_ns = elapsed * 1000000000

#     return (
#         " " * max(1, MIN_TIME_PADDING - len(message))
#         + prefix
#         + f" {round_to_resolution(elapsed_ns, resolution_ns):<4} ns"
#     )


# @dataclass(eq=True, frozen=True)
# class StopEvent:
#     """An event guarding the end of the trace."""

#     exception: Exception | None


# @dataclass(eq=True, frozen=True)
# class TracedEvent(abc.ABC):
#     """An event in a trace, having both a current and previous timestamp."""

#     @classmethod
#     @abc.abstractmethod
#     def from_frame(
#         cls,
#         frame: types.FrameType,
#         arg: Any,
#     ) -> "TracedEvent": ...


# @dataclass(eq=True, frozen=True)
# class CallEvent(TracedEvent):
#     """A call event in a trace."""

#     name: str
#     file: Path
#     lineno: int | None

#     @classmethod
#     def get_name(cls, frame: types.FrameType) -> str:
#         """Get the name of a function."""
#         if PYTHON_VERSION.minor <= 10:
#             return frame.f_code.co_name
#         return frame.f_code.co_qualname

#     @classmethod
#     def from_frame(
#         cls,
#         frame: types.FrameType,
#         arg: None,
#     ) -> "CallEvent":
#         """Construct the representation from a frame object."""
#         return cls(
#             name=cls.get_name(frame),
#             file=Path(frame.f_code.co_filename),
#             lineno=frame.f_code.co_firstlineno,
#         )


# @dataclass(eq=True, frozen=True)
# class LineEvent(TracedEvent):
#     """A call event in a trace."""

#     contents: str

#     @classmethod
#     def from_frame(
#         cls,
#         frame: types.FrameType,
#         arg: None,
#     ) -> "LineEvent":
#         """Construct the representation from a frame object."""
#         try:
#             frameinfo = inspect.getframeinfo(frame)
#             code_context = frameinfo.code_context
#             assert code_context is not None
#             contents = code_context[0].strip()
#         except Exception:
#             contents = "<source not available>"

#         return cls(
#             contents=contents,
#         )


# @dataclass(eq=True, frozen=True)
# class ReturnEvent(TracedEvent):
#     """A call event in a trace."""

#     return_value: Any

#     @classmethod
#     def from_frame(
#         cls,
#         frame: types.FrameType,
#         arg: None,
#     ) -> "ReturnEvent":
#         """Construct the representation from a frame object."""
#         return cls(
#             return_value=arg
#         )


# @dataclass(eq=True, frozen=True)
# class ExceptionEvent(TracedEvent):
#     """A call event in a trace."""

#     exception: type
#     value: Exception = field(compare=False)
#     traceback: types.TracebackType | None = field(compare=False)

#     @classmethod
#     def from_frame(
#         cls,
#         frame: types.FrameType,
#         arg: tuple[type, Exception, types.TracebackType | None],
#     ) -> "ExceptionEvent":
#         """Construct the representation from a frame object."""
#         return cls(
#             exception=arg[0],
#             value=arg[1],
#             traceback=arg[2],
#         )


# @dataclass(eq=True, frozen=True)
# class OpcodeEvent(TracedEvent):
#     """A call event in a trace."""

#     lineno: int | None
#     curr_instr: bool
#     jump: bool
#     offset: int
#     opname: str
#     arg: int | None
#     argrepr: str

#     @classmethod
#     def get_opcode(cls, frame: types.FrameType) -> dis.Instruction | None:
#         """Get the current opcode name for a frame.

#         This can be found by retrieving the instruction at the correct offset in
#         the frame.
#         """
#         for instr in dis.get_instructions(frame.f_code):
#             if instr.offset == frame.f_lasti:
#                 return instr
#         return None

#     @classmethod
#     def get_lineno(cls, instruction: dis.Instruction) -> int | None:
#         """Get the line number for an instruction."""
#         if PYTHON_VERSION.minor <= 10:
#             return instruction.starts_line
#         if instruction.line_number is not None:
#             return instruction.line_number
#         return getattr(instruction.positions, "lineno", None)

#     @classmethod
#     def from_frame(
#         cls,
#         frame: types.FrameType,
#         arg: None,
#     ) -> "OpcodeEvent":
#         """Construct the representation from a frame object."""
#         instruction = cls.get_opcode(frame)
#         assert instruction is not None

#         return cls(
#             lineno=cls.get_lineno(instruction),
#             curr_instr=False,
#             jump=instruction.is_jump_target,
#             offset=instruction.offset,
#             opname=instruction.opname,
#             arg=instruction.arg,
#             argrepr=instruction.argrepr,
#         )


# EVENT_NAME_LOOKUP: dict[str, type[TracedEvent]] = {
#     "call": CallEvent,
#     "line": LineEvent,
#     "return": ReturnEvent,
#     "exception": ExceptionEvent,
#     "opcode": OpcodeEvent,
# }


# EventTimestamp: TypeAlias = tuple[str, float | None, float | None]


# @dataclass
# class BytecodeProfiler:
#     """A profiler for Python (>=3.7) bytecode."""

#     events: list[TracedEvent | StopEvent] = field(default_factory=list)
#     timestamps: list[EventTimestamp] = field(default_factory=list)
#     _prev_event_timestamp: float | None = None
#     elapsed_times: dict[TracedEvent | StopEvent, float] = field(default_factory=dict)
#     uninstrumented_time: float | None = None
#     instrumented_time: float | None = None

#     def reset(self) -> None:
#         """Reset the profiler."""
#         self.__init__()

#     def _trace__collect_event_timestamps(
#         self, frame: types.FrameType, _event: str, _arg: Any
#     ) -> Callable[..., Any] | None:
#         """Trace function which does nothing."""
#         now_timestamp = perf_counter()
#         frame.f_trace_opcodes = True

#         # TODO: Add something further to make id unique...
#         self.timestamps.append(
#             (_event, self._prev_event_timestamp, now_timestamp,)
#         )

#         self._prev_event_timestamp = perf_counter()
#         return self._trace__collect_event_timestamps

#     def _trace__collect_all_events(
#         self, frame: types.FrameType, event: str, arg: Any
#     ) -> Callable[..., Any] | None:
#         """Collect all events emitted by the function being traced."""
#         frame.f_trace_opcodes = True

#         assert event in EVENT_NAME_LOOKUP
#         self.events.append(
#             EVENT_NAME_LOOKUP[event].from_frame(
#                 frame=frame,
#                 arg=arg,
#             )
#         )

#         return self._trace__collect_all_events

#     def profile(
#         self, func: Callable[..., Any], *args: list[Any], **kwargs: dict[str, Any]
#     ) -> None:
#         """Collect all events emitted when invoking a function."""
#         if PYTHON_VERSION.minor < 10:
#             raise RuntimeError("Tracing only supported for Python versions >=3.10!")

#         old_trace = sys.gettrace()
#         gcold = gc.isenabled()
#         gc.disable()

#         exception = None
#         finish_timestamp: float | None = None
#         try:
#             # Warmup cache/specialising adaptive interpreter/JIT
#             for _ in range(FUNC_WARMUPS):
#                 # try:
#                 func(*args, **kwargs)
#                 # except Exception as _exc:
#                 #     pass

#             # Measure the time taken to run the function once without tracing
#             start_timestamp: float | None = None
#             try:
#                 start_timestamp = perf_counter()
#                 func(*args, **kwargs)
#             except Exception as _exc:
#                 pass
#             finally:
#                 finish_timestamp = perf_counter()
#                 self.uninstrumented_time = (
#                     finish_timestamp - start_timestamp
#                     if start_timestamp is not None
#                     else None
#                 )

#             # Trace the function collecting the event data to corrollate
#             try:
#                 sys.settrace(self._trace__collect_all_events)
#                 func(*args, **kwargs)
#             except Exception as exc:
#                 exception = exc
#             finally:
#                 sys.settrace(None)

#             # Trace the function collecting only the timestamps
#             start_timestamp = None
#             try:
#                 sys.settrace(self._trace__collect_event_timestamps)
#                 start_timestamp = perf_counter()
#                 func(*args, **kwargs)
#             except Exception as exc:
#                 exception = exc
#             finally:
#                 sys.settrace(None)
#                 finish_timestamp = perf_counter()
#                 self.instrumented_time = (
#                     finish_timestamp - start_timestamp
#                     if start_timestamp is not None
#                     else None
#                 )

#         finally:
#             if gcold:
#                 gc.enable()
#             sys.settrace(old_trace)

#         self.events.append(StopEvent(exception))
#         self.timestamps.append(("stop", finish_timestamp, None))

#         self.elapsed_times = self._calculate_elapsed_times()

#     def _calculate_elapsed_times(self) -> dict[TracedEvent | StopEvent, float]:
#         """Calculate the elapsed time for each event."""
#         assert len(self.events) == len(self.timestamps)


#         native_tracing_time = 0.0
#         for i in range(len(self.timestamps)):
#             if i < len(self.events) - 1:
#                 assert (start_tracing_timestamp := self.timestamps[i][2]) is not None
#                 assert (end_tracing_timestamp := self.timestamps[i+1][1]) is not None
#                 native_tracing_time += end_tracing_timestamp - start_tracing_timestamp

#         print(f"Uninstrumented = {self.uninstrumented_time * 1000000000:.1f}ns")
#         print(f"Instrumented = {self.instrumented_time * 1000000000:.1f}ns")
#         print(f"Tracing = {native_tracing_time * 1000000000:.1f}ns")
#         print(
#             f"Uninstrumented + Tracing = {(self.uninstrumented_time + native_tracing_time) * 1000000000:.1f}ns"
#         )
#         print(f"Non-tracing = {(self.instrumented_time - native_tracing_time) * 1000000000:.1f}ns")

#         average_nontracing_time = (self.instrumented_time - native_tracing_time) / len(self.events)
#         print(f"Offset = {average_nontracing_time * 1000000000:.1f}ns")

#         elapsed_times: dict[TracedEvent | StopEvent, float] = {}
#         for i, timestamp in enumerate(self.timestamps[:-1]):
#             if i == 0:
#                 continue
#             assert (start_execution_timestamp := timestamp[1]) is not None
#             assert (end_execution_timestamp := timestamp[2]) is not None
#             elapsed_times[self.events[i-1]] = (
#                 end_execution_timestamp
#                 - start_execution_timestamp
#                 - average_nontracing_time
#             )
#         elapsed_times[self.events[-1]] = 0

#         return elapsed_times

#     @classmethod
#     def print_events(
#         cls,
#         events: list[Any],
#         elapsed_times: dict[Any, float],
#         trace_name: str,
#         indent_size: int = 4,
#     ) -> None:
#         """Print a string representation of the traced events."""

#         # Print the events with this timing information
#         print(f"//// Trace of `{trace_name}` :")

#         indent = -indent_size
#         for event in events:
#             # print(event, elapsed_times[event])
#             if isinstance(event, CallEvent):
#                 indent += indent_size
#                 contents = f" {event.file.stem}:{event.lineno} `{event.name}` "
#                 line = f"\n{' ' * indent}// =={contents:{'='}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}=="
#                 print(line + " " + padded_time(line, "==", elapsed_times[event]))
#             elif isinstance(event, LineEvent):
#                 line = f"{' ' * indent}// >>> {event.contents}"
#                 print(line + padded_time(line, "##", elapsed_times[event]))
#             elif isinstance(event, ReturnEvent):
#                 print(f"{' ' * (indent)}// {'=' * FUNCTION_DELIMITER_LENGTH}\n")
#                 indent -= indent_size
#             elif isinstance(event, ExceptionEvent):
#                 contents = f" `{event.exception.__qualname__}` "
#                 line = f"\n{' ' * indent}// !!{contents:{'!'}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}!!"
#                 print(line + " " + padded_time(line, "==", elapsed_times[event]))
#             if isinstance(event, OpcodeEvent):
#                 # TODO: Lift back into dataclasses
#                 curr_instr = "-->" if event.curr_instr else ""
#                 jump = ">> " if event.jump else ""
#                 line_no = event.lineno if event.lineno is not None else ""
#                 arg = event.arg if event.arg is not None else ""
#                 line = (
#                     " " * indent
#                     + f"{line_no:>3} {curr_instr:>3} {jump:>3} {event.offset:<3} "
#                     + f"{event.opname:<20} {arg:<3} {'(' + event.argrepr + ')':<20} "
#                 )
#                 print(line + padded_time(line, "//", elapsed_times[event]))


# def print_bytecode(
#     func: Callable[[], Any],
#     *args: list[Any],
#     **kwargs: dict[str, Any],
# ) -> None:
#     """Print the bytecode executed when running a function."""
#     profiler = BytecodeProfiler()
#     profiler.profile(func, *args, **kwargs)
#     BytecodeProfiler.print_events(
#         profiler.events,
#         profiler.elapsed_times,
#         trace_name=func.__qualname__
#     )


# if __name__ == "__main__":

#     def inner_function(x: int | str | float):
#         assert x

#     def raise_exception():
#         raise ValueError("Help")

#     def example_function():
#         inner_function(1)
#         inner_function("Hello")
#         x = perf_counter()
#         # raise_exception()
#         inner_function(5.0)

#     def go():
#         example_function()

#     print_bytecode(go)
