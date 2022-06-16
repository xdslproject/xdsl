import re

import argparse
import sys
import os
from xdsl.printer import *


#try:
#    print(1 / 0)
#except Exception:
#    log.exception("unable print!")


def get_frame(num, e: DiagnosticException):

    trace = traceback.format_exc().splitlines()
    length = num if num < len(trace) else len(trace)
#    traceback.extract_stack()
    #for i in range(0-length,0):

    exc_type, exc_value, exc_traceback = sys.exc_info()
    # only doing last frame for now,
    # TODO: arbitrary frame

    locals = sys.exc_info()[2].tb_next.tb_frame.f_locals
#    print(locals)
    extract_frame = traceback.format_tb(exc_traceback)[-1]
#    print(extract_frame)

#    print([i.split(', ') for i in extract_frame.split('\n')])
    f = [i.split(', ') for i in extract_frame.split('\n')][0]
        # file | line | location
    filename  = re.search(r'\"(.*?)\"', f[0]).group(1)
    local_var = inspect.trace()[-1][0].f_locals
    line_num  = f[1].split(' ')[-1]

    exc_name = repr(exc_type.mro()[0]).rstrip('>').lstrip('<').split(' ')[-1]
    return  filename, line_num, exc_name, local_var, extract_code(filename, int(line_num)),

   
def extract_code(filename, line_num: int):
    """
    Extract info from given frame. 
    Returns the code snippet.
    """
    code: str = ""

    with open(filename) as fp:
        for i, line in enumerate(fp):
            if i >= line_num - 3 and i <= line_num + 3:
                code+=line
    return code

def diagnostic(num, e: DiagnosticException) -> None:
    filename, line_num, exc_name, local_var, code = get_frame(num, e)

    print(exc_name,": ", e, "\n")

    print("filename: ", filename)
    print("line: ", line_num)
    print("code: \n", code, "\n")

    print("local variable(s): \n", local_var)
    exit(0)

