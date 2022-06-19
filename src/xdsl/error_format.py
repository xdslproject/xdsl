import os
import sys
from types import TracebackType


class Colors:

    class format:
        reset = '\033[0m'
        bold = '\033[01m'
        underline = '\033[04m'

    class fg:
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        cyan = '\033[36m'


class Frame:
    """
    This class is a traceback wrapper of a given raised error 
    """
    e: Exception
    extra_lines: int

    def __init__(self, e: Exception, extra_lines: int):
        self.e = e
        self.extra_lines = extra_lines

    def get_frame(self, num: int):
        """
        Extracting the frame from the traceback.
        With given exception
        """
        _, _, exc_traceback = sys.exc_info()

        stack: list[TracebackType] = []
        tb = exc_traceback
        while tb:
            stack.append(tb)
            tb = tb.tb_next
        stack = stack[-num:] if num else []

        for tb in stack:
            frame = tb.tb_frame
            code = frame.f_code

            filename = code.co_filename
            line_num = frame.f_lineno
            exc_name = code.co_name
            local_var = frame.f_locals

            yield filename, line_num, exc_name, local_var, self.extract_code(
                filename, line_num)

    def extract_code(self, filename: str, line_num: int):
        """
        Extract the code from given frame. 
        returns a string of the code
        """
        code: str = "\n"

        with open(filename) as fp:
            for i, line in enumerate(fp):
                if line_num - self.extra_lines <= i <= line_num + self.extra_lines:
                    if i == line_num - 1:
                        code += Colors.format.reset + Colors.fg.red + str(
                            i + 1) + line
                        code += ' ' * len(
                            str(i + 1)) + '^' * (len(line) - 1) + '\n'
                    else:
                        code += Colors.fg.orange + str(i + 1) + line

        return code + "\n"

    def verbose(self, num: int) -> None:
        """
        Output the verbose diagnostic message
        """
        for filename, line_num, exc_name, local_var, code in self.get_frame(
                num):

            print(Colors.fg.red, exc_name, ": ", self.e, "\n")

            print(Colors.fg.green, "filename: ", end='')
            print(Colors.fg.orange, filename)

            print(Colors.fg.green, "line: ", end='')
            print(Colors.fg.orange, line_num)

            print(Colors.fg.green, "code: \n", end='')
            print(Colors.fg.orange, "```\n", code, "```\n")

            print(Colors.fg.green, "local variable(s): \n")

            for k, v in local_var.items():
                print(Colors.fg.cyan, k, ": \n", end="")
                print(Colors.format.reset, v, "\n")

            print(Colors.format.reset, "")
            print('─' * os.get_terminal_size().columns)
