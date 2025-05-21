"""A bytecode performance profiler."""
# TODO: When in print mode disable warmup and don't measure timestamps
# TODO: Investigate specialising adaptive interpreter
# TODO: `OpcodeEvent` could just store the instruction and make existing variables properties?
# TODO: Split out into separate project
# TODO: Make fun textual UI

import abc
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
from time import perf_counter
from typing import Any, TypeAlias

FUNC_WARMUPS = 1500
FUNC_REPEATS = 1500
FUNCTION_DELIMITER_LENGTH = 45
MIN_TIME_PADDING = 100

PYTHON_VERSION = sys.version_info
PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution
NS_PER_S = 1000000000


def round_to_ns_resolution(
    measurement: float, resolution: float = PERF_COUNTER_RESOLUTION
) -> float:
    """Round a floating point value to a resolution."""
    if resolution == 0:
        return measurement * NS_PER_S

    factor = 10 ** math.floor(math.log10(resolution * NS_PER_S))
    return round(measurement * NS_PER_S / factor) * factor


def padded_time(message: str, prefix: str, elapsed: float) -> str:
    """Get a padded string containing the time for a trace event."""
    return (
        " " * max(1, MIN_TIME_PADDING - len(message))
        + prefix
        + f" {round_to_ns_resolution(elapsed):<4} ns"
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

        # TODO: `OpcodeEvent` could just store the instruction and make these properties?
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


EventTimestamp: TypeAlias = tuple[float | None, float | None]


@dataclass
class BytecodeProfiler:
    """A profiler for Python (>=3.7) bytecode."""

    num_warmups: int = FUNC_WARMUPS
    num_repeats: int = FUNC_REPEATS
    debug: bool = True

    _events: list[TracedEvent | StopEvent] = field(default_factory=list)
    _timestamps: list[EventTimestamp] = field(default_factory=list)
    _uninstrumented_time: float | None = None
    _instrumented_time: float | None = None
    _prev_event_timestamp: float | None = None

    _opcode_overhead: float = 0.0

    @property
    def events(self) -> list[TracedEvent | StopEvent]:
        return self._events

    @property
    def uninstrumented_time(self) -> float | None:
        return self._uninstrumented_time

    @property
    def instrumented_time(self) -> float | None:
        return self._instrumented_time

    def _reset(self) -> None:
        """Reset the profiler."""
        self._events = []
        self._timestamps = []
        self._prev_event_timestamp = None
        self._uninstrumented_time = None
        self._instrumented_time = None

    def _trace__collect_event_timestamps(
        self, frame: types.FrameType, event: str, _arg: Any
    ) -> Callable[..., Any] | None:
        """Trace function which does nothing."""
        now_timestamp = perf_counter()
        frame.f_trace_lines = False
        frame.f_trace_opcodes = True

        if event == "opcode":
            self._timestamps.append(
                (
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
        self._events.append(
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

        self._reset()

        old_trace = sys.gettrace()
        gcold = gc.isenabled()
        gc.disable()

        exception = None
        try:
            # Warmup cache/specialising adaptive interpreter/JIT
            for _ in range(self.num_warmups):
                try:
                    func(*args, **kwargs)
                except Exception as _exc:
                    pass

            # Measure the time taken to run the function once without tracing
            self._uninstrumented_time = 0
            for _ in range(self.num_repeats):
                start_timestamp: float | None = None
                try:
                    start_timestamp = perf_counter()
                    func(*args, **kwargs)
                    finish_timestamp = perf_counter()
                except Exception as _exc:
                    finish_timestamp = perf_counter()
                assert start_timestamp is not None
                self._uninstrumented_time += finish_timestamp - start_timestamp
            self._uninstrumented_time /= self.num_repeats

            # Trace the function collecting the event data to corrollate
            try:
                sys.settrace(self._trace__collect_all_events)
                func(*args, **kwargs)
            except Exception as exc:
                exception = exc
            finally:
                sys.settrace(None)
            self._events.append(StopEvent(exception))

            # Trace the function collecting only the timestamps
            instrumented_time_repeats: list[float] = []
            for _ in range(self.num_repeats):
                start_timestamp: float | None = None
                try:
                    sys.settrace(self._trace__collect_event_timestamps)
                    start_timestamp = perf_counter()
                    func(*args, **kwargs)
                    finish_timestamp = perf_counter()
                except Exception as _exc:
                    finish_timestamp = perf_counter()
                finally:
                    sys.settrace(None)
                self._timestamps.append((self._prev_event_timestamp, finish_timestamp))
                assert start_timestamp is not None
                instrumented_time_repeats.append(finish_timestamp - start_timestamp)
            self._instrumented_time = statistics.mean(instrumented_time_repeats)

        finally:
            if gcold:
                gc.enable()
            sys.settrace(old_trace)
            opcode_events = [
                e
                for e in self._events
                if isinstance(e, OpcodeEvent) or isinstance(e, StopEvent)
            ]
            assert len(opcode_events) * self.num_repeats == len(self._timestamps)

    @classmethod
    def get_nop_sled(cls, size: int) -> Callable[..., None]:
        """Get a function which is a nop sled for baseline profiling."""
        many_passes = "\n    ".join("pass" for _ in range(size))
        namespace: dict[str, Any] = {}
        exec(f"def nop_sled():\n    {many_passes}", {}, namespace)
        return namespace["nop_sled"]

    def calibrate_opcode_overhead(self, num_opcodes: int = 1000) -> None:
        """Calibrate opcode tracing overhead by tracing many NOP opcodes."""

        nop_sled = self.get_nop_sled(num_opcodes)
        self.profile(nop_sled)
        assert self._uninstrumented_time is not None
        assert self._instrumented_time is not None
        assert len(self._events)
        assert isinstance(self._events[-1], StopEvent)
        assert self._events[-1].exception is None

        elapsed_times = self.calculate_elapsed_times()
        self._opcode_overhead = (sum(elapsed_times) - self._uninstrumented_time) / len(
            elapsed_times
        )

        self._reset()

    def calculate_elapsed_times(self) -> list[float]:
        """Calculate the elapsed time for each event."""
        timestamp_repeats: list[list[EventTimestamp]] = []
        opcode_events = [
            e
            for e in self._events
            if isinstance(e, OpcodeEvent) or isinstance(e, StopEvent)
        ]
        for i in range(0, len(self._timestamps), len(opcode_events)):
            timestamp_repeats.append(self._timestamps[i : i + len(opcode_events)])

        elapsed_time_repeats: list[list[float]] = []
        for timestamps in timestamp_repeats:
            elapsed_times: list[float] = [0.0 for _ in range(len(timestamps))]
            for i, timestamp in enumerate(timestamps):
                if i == 0:
                    continue
                assert (start_execution_timestamp := timestamp[0]) is not None
                assert (end_execution_timestamp := timestamp[1]) is not None
                elapsed_times[i - 1] = (
                    end_execution_timestamp
                    - start_execution_timestamp
                    - self._opcode_overhead
                )
            elapsed_time_repeats.append(elapsed_times)

        elapsed_times = [
            # statistics.mean(event_elapsed_times)
            min(event_elapsed_times)
            for event_elapsed_times in zip(*elapsed_time_repeats)
        ]

        if self.debug:
            assert self._uninstrumented_time is not None
            assert self._instrumented_time is not None
            print("===========================================")
            print(f" Events recorded: {len(self.events)}")
            print(f" Opcodes recorded: {len(elapsed_times)}")
            print(f" Uninstrumented: {self._uninstrumented_time * NS_PER_S:.2f} ns")
            print(
                f" Uninstrumented per opcode: {(self._uninstrumented_time / len(elapsed_times)) * NS_PER_S:.2f} ns/op"
            )
            print(f" Instrumented: {self._instrumented_time * NS_PER_S:.2f} ns")
            print(
                f" Instrumented per opcode: {(self._instrumented_time / len(elapsed_times)) * NS_PER_S:.2f} ns/op"
            )
            print(f" Elapsed: {sum(elapsed_times) * NS_PER_S:.2f} ns")
            print(
                f" Elapsed per opcode: {statistics.mean(elapsed_times) * NS_PER_S:.2f} ns/op"
            )
            print(f" Opcode overhead: {self._opcode_overhead * NS_PER_S:.2f} ns/op")
            tracing_overhead = (sum(elapsed_times) - self._uninstrumented_time) / len(
                elapsed_times
            )
            print(
                f" Tracing overhead per opcode: {tracing_overhead * NS_PER_S:.2f} ns/op"
            )
            print("===========================================", end="\n\n")

        return elapsed_times

    def calculate_trace_times(self) -> list[float]:
        """Calculate the trace time for each event."""
        elapsed_times = self.calculate_elapsed_times()
        assert self.uninstrumented_time is not None
        scaling_factor = self.uninstrumented_time / sum(elapsed_times)
        return [elapsed_time * scaling_factor for elapsed_time in elapsed_times]

    @classmethod
    def print_events(
        cls,
        events: list[Any],
        trace_name: str,
        elapsed_times: list[float] | None = None,
        indent_size: int = 4,
        debug: bool = True,
    ) -> None:
        """Print a string representation of the traced events."""
        if elapsed_times is not None:
            assert len(
                [
                    e
                    for e in events
                    if isinstance(e, OpcodeEvent) or isinstance(e, StopEvent)
                ]
            ) == len(elapsed_times)

        print(f"//// Trace of `{trace_name}` :")
        indent = -indent_size
        elapsed_time_index = 0
        for i, event in enumerate(events):
            if isinstance(event, CallEvent):
                indent += indent_size
                contents = f" {event.file.stem}:{event.lineno} `{event.name}` "
                print(
                    f"\n{' ' * indent}// =={contents:{'='}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}=="
                )
            elif isinstance(event, LineEvent):
                print(f"{' ' * indent}// >>> {event.contents}")
            elif isinstance(event, ReturnEvent):
                print(f"{' ' * (indent)}// {'=' * FUNCTION_DELIMITER_LENGTH}\n")
                indent -= indent_size
            elif isinstance(event, ExceptionEvent):
                contents = f" `{event.exception.__qualname__}` "
                print(
                    f"\n{' ' * indent}// !!{contents:{'!'}{'^'}{FUNCTION_DELIMITER_LENGTH - 4}}!!"
                )
            if isinstance(event, OpcodeEvent):
                curr_instr = "-->" if event.curr_instr else ""
                jump = ">> " if event.jump else ""
                line_no = event.lineno if event.lineno is not None else ""
                arg = event.arg if event.arg is not None else ""
                line = (
                    " " * indent
                    + f"{line_no:>3} {curr_instr:>3} {jump:>3} {event.offset:<3} "
                    + f"{event.opname:<20} {arg:<3} {'(' + event.argrepr + ')':<20} "
                )
                if elapsed_times is not None:
                    line += padded_time(line, "//", elapsed_times[elapsed_time_index])
                    elapsed_time_index += 1
                print(line)


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
    profiler.profile(func, *args, **kwargs)
    BytecodeProfiler.print_events(
        events=profiler.events,
        trace_name=func.__qualname__,
        elapsed_times=None,
    )


def profile_bytecode(
    func: Callable[[], Any],
    *args: list[Any],
    **kwargs: dict[str, Any],
) -> None:
    """Profile the bytecode executed when running a function.

    The function must be pure, as it is run many times to get various peices of
    data.
    """
    profiler = BytecodeProfiler()
    profiler.profile(func, *args, **kwargs)
    elapsed_times = profiler.calculate_trace_times()

    BytecodeProfiler.print_events(
        events=profiler.events,
        trace_name=func.__qualname__,
        elapsed_times=elapsed_times,
    )


if __name__ == "__main__":

    def inner_function(x: int | str | float):
        assert x

    def raise_exception():
        raise ValueError("Help")

    def example_function():
        pass
        pass
        inner_function(1)
        inner_function("Hello")
        _x = perf_counter()
        pass
        raise_exception()
        inner_function(5.0)

    nop_sled = BytecodeProfiler.get_nop_sled(50)

    def go():
        example_function()

    profile_bytecode(go)
