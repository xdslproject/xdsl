import re
import sys
import os
import traceback
import inspect


def get_frame(extra_lines, num: int, e: Exception):
    """
    Extracting the frame from the traceback.
    Given 
    """

    trace = traceback.format_exc().splitlines()
    exc_type, _, exc_traceback = sys.exc_info()
    locals = sys.exc_info()[2].tb_next.tb_frame.f_locals

    full_tb = traceback.format_tb(exc_traceback)

    start = 0 if num >= len(full_tb) else len(full_tb) - num

    # last frame will be extracted
    for i in range(start, len(full_tb)):

        extract_frame = traceback.format_tb(exc_traceback)[i]
        f = [j.split(', ') for j in extract_frame.split('\n')][0]
        # file | line | location
        filename = re.search(r'\"(.*?)\"', f[0]).group(1)
        local_var = inspect.trace()[i][0].f_locals
        line_num = f[1].split(' ')[-1]

        exc_name = repr(
            exc_type.mro()[0]).rstrip('>').lstrip('<').split(' ')[-1]

        yield filename, line_num, exc_name, local_var, extract_code(
            filename, extra_lines, int(line_num)),


def extract_code(filename, extra_lines: int, line_num: int):
    """
    Extract info from given frame. 
    Returns the code snippet.
    """
    code: str = "\n"

    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i >= line_num - extra_lines and i <= line_num + extra_lines:
                if i == line_num - 1:
                    code += Colors.format.reset + Colors.fg.red + str(
                        i + 1) + line  #+'\033[0m'
                else:
                    code += Colors.fg.orange + str(i + 1) + line

    return code + "\n"


# Python program to print
# colored text and background
class Colors:

    class format:
        reset = '\033[0m'
        bold = '\033[01m'
        underline = '\033[04m'

    class fg:
        #		black='\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        cyan = '\033[36m'


def diagnostic(num: int, e: Exception) -> None:
    for filename, line_num, exc_name, local_var, code in get_frame(3, num, e):

        print(Colors.fg.red, exc_name, ": ", e, "\n")

        print(Colors.fg.green, "filename: ",end='')
        print(Colors.fg.orange,filename)
        print(Colors.fg.green, "line: ",end='')
        print(Colors.fg.orange, line_num)

        print(Colors.fg.green, "code: \n", end='')
        print(Colors.fg.orange, "```\n",code, "```\n")
        #    p = Printer()

        print(Colors.fg.green, "local variable(s): \n\n")
        for k, v in list(local_var.items()):
            print(Colors.fg.cyan, k, ": \n", end="")
            print(Colors.format.reset, v, "\n")

        #    p.print_op(list(local_var.values())[0])
        #    print(type(list(local_var.values())[0]))
        print(Colors.format.reset, "")
        print('─' * os.get_terminal_size().columns)  # U+2501, Box Drawings Heavy Horizontal
    exit(0)
