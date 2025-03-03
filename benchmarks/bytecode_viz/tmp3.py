import dis
import inspect
import linecache
import sys


class DetailedBytecodeTracer:
    def __init__(self):
        self.instructions = []
        self.target_function_name = None
        self.depth = 0

    def trace_function(self, func):
        """Decorator to trace a function's bytecode execution."""

        def wrapper(*args, **kwargs):
            self.target_function_name = func.__name__
            self.instructions = []
            self.depth = 0

            # Create a specialized frame evaluation function
            def trace_frames(frame, event, arg):
                # Only trace our target function
                if frame.f_code.co_name == self.target_function_name:
                    if event == "call":
                        self.depth += 1

                        # Get arguments for this call
                        args_info = inspect.getargvalues(frame)
                        args_str = ", ".join(
                            f"{arg}={args_info.locals[arg]}" for arg in args_info.args
                        )

                        # Add entry for this recursive call with proper depth
                        indent = "  " * (self.depth - 1)
                        self.instructions.append(
                            f"\n{indent}--- Recursion level {self.depth} with {args_str} ---"
                        )

                        # Show source code for the current line
                        current_line = linecache.getline(
                            frame.f_code.co_filename, frame.f_lineno
                        ).strip()
                        self.instructions.append(
                            f"{indent}Line {frame.f_lineno}: {current_line}"
                        )

                        # Show bytecode for this function call
                        for instr in dis.get_instructions(frame.f_code):
                            self.instructions.append(
                                f"{indent}  {instr.offset}: {instr.opname} {instr.argrepr if instr.arg is not None else ''}"
                            )

                    elif event == "line":
                        # Add indentation based on recursion depth
                        indent = "  " * (self.depth - 1)

                        # Show the line being executed
                        current_line = linecache.getline(
                            frame.f_code.co_filename, frame.f_lineno
                        ).strip()
                        self.instructions.append(
                            f"{indent}Executing line {frame.f_lineno}: {current_line}"
                        )

                        # Show the current bytecode offset
                        self.instructions.append(
                            f"{indent}  Current bytecode offset: {frame.f_lasti}"
                        )

                        # Find the instructions for this line
                        found_line = False
                        for instr in dis.get_instructions(frame.f_code):
                            if instr.starts_line == frame.f_lineno:
                                found_line = True
                            if found_line and (
                                instr.starts_line is None
                                or instr.starts_line == frame.f_lineno
                            ):
                                self.instructions.append(
                                    f"{indent}  > {instr.offset}: {instr.opname} {instr.argrepr if instr.arg is not None else ''}"
                                )
                            elif found_line:
                                break

                    elif event == "return":
                        # Add indentation based on recursion depth
                        indent = "  " * (self.depth - 1)

                        # Show the return value
                        self.instructions.append(f"{indent}Returning value: {arg}")
                        self.instructions.append(
                            f"{indent}--- End recursion level {self.depth} ---\n"
                        )
                        self.depth -= 1

                return trace_frames

            # Set the trace function
            old_trace = sys.settrace(trace_frames)

            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the original trace function
                sys.settrace(old_trace)

        return wrapper

    def get_instructions(self):
        """Return the list of traced instructions."""
        return self.instructions


class ConciseBytecodeTracer:
    """A simpler tracer that just shows the executed bytecode in sequence."""

    def __init__(self):
        self.instructions = []
        self.target_function_name = None
        self.depth = 0

    def trace_function(self, func):
        """Decorator to trace a function's bytecode execution."""

        def wrapper(*args, **kwargs):
            self.target_function_name = func.__name__
            self.instructions = []
            self.depth = 0

            # Create a specialized frame evaluation function
            def trace_frames(frame, event, arg):
                if frame.f_code.co_name == self.target_function_name:
                    if event == "call":
                        self.depth += 1
                        # Get arguments info
                        args_info = inspect.getargvalues(frame)
                        args_str = ", ".join(
                            f"{arg}={args_info.locals[arg]}" for arg in args_info.args
                        )
                        indent = "  " * (self.depth - 1)

                        self.instructions.append(
                            f"{indent}CALL {self.target_function_name}({args_str})"
                        )

                    elif event == "line":
                        indent = "  " * (self.depth - 1)

                        # Find the instruction we're about to execute
                        for instr in dis.get_instructions(frame.f_code):
                            if instr.starts_line == frame.f_lineno:
                                self.instructions.append(
                                    f"{indent}{instr.offset}: {instr.opname} "
                                    f"{instr.argrepr if instr.arg is not None else ''}"
                                )
                                break

                    elif event == "return":
                        indent = "  " * (self.depth - 1)
                        self.instructions.append(f"{indent}RETURN {arg}")
                        self.depth -= 1

                return trace_frames

            # Set the trace function
            old_trace = sys.settrace(trace_frames)

            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the original trace function
                sys.settrace(old_trace)

        return wrapper

    def get_instructions(self):
        """Return the list of traced instructions."""
        return self.instructions


# Example usage
if __name__ == "__main__":
    # Use the DetailedBytecodeTracer
    tracer = DetailedBytecodeTracer()

    @tracer.trace_function
    def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n - 1)

    # Execute factorial with n=3 (should have 3 recursive levels)
    print("Calculating factorial(3)...")
    result = factorial(3)
    print(f"Result: {result}")

    # Print traced bytecode instructions
    print("\nDetailed bytecode instructions executed:")
    for instruction in tracer.get_instructions():
        print(instruction)

    # Alternative: Use the concise tracer for a cleaner view
    print("\n\nConcise bytecode trace:")
    concise_tracer = ConciseBytecodeTracer()

    @concise_tracer.trace_function
    def factorial(n):
        if n <= 1:
            return 1
        else:
            return n * factorial(n - 1)

    result = factorial(3)
    for instruction in concise_tracer.get_instructions():
        print(instruction)
