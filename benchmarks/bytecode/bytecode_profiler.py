import abc
import dis
import gc
import inspect
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PYTHON_VERSION = sys.version_info
PERF_COUNTER_RESOLUTION = time.get_clock_info("perf_counter").resolution


@dataclass
class StopEvent:
    """An event with a previous timestamp."""

    prev_timestamp: float
    exception: Exception | None


@dataclass
class _LogMessage:
    """An event with a previous timestamp."""

    message: Any


@dataclass
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


@dataclass
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


@dataclass
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


@dataclass
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


@dataclass
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


@dataclass
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
            return None
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


@dataclass
class BytecodeProfiler:
    """A profiler for Python (>=3.7) bytecode."""

    _events: list[TracedEvent | StopEvent | _LogMessage] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    _prev_timestamp: float | None = None

    def reset(self) -> None:
        """Reset the profiler."""
        self.__init__()

    def trace__collect_all_events(
        self, frame: types.FrameType, event: str, arg: Any
    ) -> Callable[..., Any] | None:
        """Collect all events emitted by the function being traced."""
        now_timestamp = time.perf_counter()
        frame.f_trace_opcodes = True

        assert self._prev_timestamp is not None
        assert event in EVENT_NAME_LOOKUP
        self._events.append(
            EVENT_NAME_LOOKUP[event].from_frame(
                frame=frame,
                arg=arg,
                prev_timestamp=self._prev_timestamp,
                now_timestamp=now_timestamp,
            )
        )

        self._prev_timestamp = time.perf_counter()
        return self.trace__collect_all_events

    def collect_all_events(
        self, func: Callable[..., Any], *args: list[Any], **kwargs: dict[str, Any]
    ) -> None:
        """Collect all events emitted when invoking a function."""
        gcold = gc.isenabled()
        gc.disable()
        old_trace = sys.gettrace()
        exception = None

        try:
            sys.settrace(self.trace__collect_all_events)
            try:
                self._prev_timestamp = time.perf_counter()
                func(*args, **kwargs)
            except Exception as _exc:
                exception = _exc
            finally:
                finish_timestamp = time.perf_counter()

        finally:
            sys.settrace(old_trace)
            if gcold:
                gc.enable()

        self._events.append(StopEvent(finish_timestamp, exception))

    def profile_function(
        self,
        func: Callable[[], Any],
        *args: list[Any],
        **kwargs: dict[str, Any],
    ) -> None:
        self.reset()
        self.collect_all_events(func, *args, **kwargs)
        for event in self._events:
            print(event)


def print_bytecode(func: Callable[[], Any]) -> None:
    """Print the bytecode executed when running a function."""
    profiler = BytecodeProfiler()
    profiler.profile_function(func)


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
