import re


def parse_stack_trace(stack_trace):
    trace_pattern = re.compile(r'"(.+)", line (\d+), in (.+)')
    error_pattern = re.compile(r"^(.+Error): (.+)$", re.MULTILINE)

    trace_matches = trace_pattern.findall(stack_trace)
    error_match = error_pattern.search(stack_trace)

    # trace_list = [(func_name, file_name, int(line_num)) for file_name, line_num, func_name in trace_matches]
    trace_list = [
        {"func_name": func_name, "file_name": file_name, "line_num": int(line_num)}
        for file_name, line_num, func_name in trace_matches
    ]
    # final_error = (error_match.group(1), error_match.group(2)) if error_match else None
    final_error = (
        {"error_type": error_match.group(1), "error_msg": error_match.group(2)}
        if error_match
        else None
    )

    return trace_list, final_error
