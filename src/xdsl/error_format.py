import os
import sys
from types import TracebackType
import traceback
import inspect


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
    Traceback wrapper of a given raised error.
    Wraps the informations in a better format
    """
    e: Exception

    def __init__(self, e):
        self.e = e

    def get_frame(self, num: int):
        """
        Extracting the frame from the traceback.
        With given number of frames from the traceback
        """
        _, _, exc_traceback = sys.exc_info()
        stack: list[TracebackType] = []
        tb = exc_traceback
        # extract stack from the traceback
        while tb:
            stack.append(tb)
            tb = tb.tb_next
        stack = stack[-num:]

        for tb in stack:
            # for each trace yield infos needed
            frame = tb.tb_frame
            code = frame.f_code
            filename = code.co_filename
            line_num = frame.f_lineno
            exc_name = code.co_name
            local_var = frame.f_locals

            code_source = inspect.getsource(code).splitlines(True)
            first_line = code.co_firstlineno

            yield filename, line_num, exc_name, local_var, self.extract_code(
                filename, first_line, line_num, code_source)

    def extract_code(self, filename, first_line, line_num, code) -> str:
        """
        Extract the code from given frame.
        By matching the line of the exception 
        it adds the carret to that line.

        returns a string of the code snippet of the function with carret indicator
        """
        carret: str = "\n"
        code.append("\n")

        with open(filename) as fp:
            for i, line in enumerate(fp):

                # different cases because the first line of the code
                # has a space in the front
                # this goes for all the frame attr

                if i == first_line:
                    code[i - first_line] = ' ' + str(i) + code[i - first_line]
                elif first_line < i < len(code) + first_line:
                    # get the length before adding anything to the line
                    line_length = len(code[i - first_line])

                    # adding numbers to the line
                    code[i - first_line] = ' ' + str(i) + code[i - first_line]

                    error_line = code[i - first_line]

                    if i == line_num:
                        # adding carret and red to the exception line
                        colored_error = Colors.format.reset + Colors.fg.red + error_line
                        carret = ' ' * (len(str(i)) + 1) + \
                                 '^' * (line_length - 1) + '\n'
                        # replacing the line with the formatted one
                        code[i - first_line] = colored_error \
                                                + carret \
                                                + Colors.fg.orange

        return "".join(code)

    def verbose(self, num: int) -> None:
        """
        Output the verbose diagnostic message
        Number of output frame is dependent on `num`:
            if num > 0: last num frames
            if num == 0: all frames
            if num < 0: drop first num of frames, get all the remaining frames
        """
        for filename, line_num, exc_name, local_var, code in self.get_frame(
                num):
            # prints out the info by frames
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
