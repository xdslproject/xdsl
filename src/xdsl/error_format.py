from sys import exc_info
from os import get_terminal_size
from types import TracebackType
from inspect import getsource


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


def get_frame(num: int):
    """
    Extracting the frame from the traceback.
    With given number of frames from the traceback
    """
    _, _, exc_traceback = exc_info()
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

        code_source = getsource(code).splitlines()
        first_line = code.co_firstlineno

        yield filename, line_num, exc_name, local_var, extract_code(
            first_line, line_num, code_source)


def extract_code(first_line: int, line_num: int, code: list[str]) -> str:
    """
    Extract the code from given frame.
    By matching the line of the exception 
    it adds the carret to that line.

    returns a string of the code snippet of the function with carret indicator
    """
    last_line: int = first_line + len(code)
    assert first_line <= line_num < first_line + len(code)
    width: int = len(str(last_line))  # calculate width for line numbers
    for i in range(first_line, last_line):
        j: int = i - first_line
        # add line number
        code[j] = f"{str(i):{width}}{code[j]}"
        if i == line_num:
            # highlight the error line and add carret
            code[j] = f"""\
{Colors.format.reset}{Colors.fg.red}{code[j]}
{" " * width}{"^" * (len(code[j]) - width)}{Colors.fg.orange}\
"""
    return "\n".join(code) + "\n"


def verbose(e: Exception, num: int) -> str:
    """
    Output the verbose diagnostic message
    Number of output frame is dependent on `num`:
        if num > 0: last num frames
        if num == 0: all frames
        if num < 0: drop first num of frames, get all the remaining frames
    """
    debug_str: str = ""
    for filename, line_num, exc_name, local_var, code in get_frame(num):
        # prints out the info by frames
        debug_str += Colors.fg.red + exc_name + ": " + str(e) + "\n"

        debug_str += Colors.fg.green + "filename: "
        debug_str += Colors.fg.orange + filename + "\n "

        debug_str += Colors.fg.green + "line: "
        debug_str += Colors.fg.orange + str(line_num) + "\n "

        debug_str += Colors.fg.green + "code: \n"
        debug_str += Colors.fg.orange + "```\n" + code + "```\n"

        debug_str += Colors.fg.green + "local variable(s): \n"

        for k, v in local_var.items():
            debug_str += Colors.fg.cyan + str(k) + ": \n"
            debug_str += Colors.format.reset + str(v) + "\n"
        debug_str += Colors.format.reset + "\n"
        debug_str += '─' * get_terminal_size().columns + "\n"

    return debug_str
