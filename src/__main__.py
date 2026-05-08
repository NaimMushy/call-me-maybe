import numpy as np
import json
import re
from .llm_sdk import Small_LLM_Model


def get_parameters(slm: Small_LLM_Model, prompt: str, function: dict[str, str | dict[str, str]]) -> list[str | float]:

    llm_prompt = f"""You are a function calling assistant. \
    Your job is to return the parameters of the function in the \
    prompt in a tupple format

    function:
        name: {function['name']}
        description: {function['description']}
        parameters: {function['parameters']}

    Rules:
        -Ouput the parameters in a tuple format inside parenthesis \
        -The parameters must have the same type as in the function \
    parameters
        - If the parameter is a string, do not generate commas between the characters \
        - The output format for each parameter should be : "<parameter_name>: <parameter_value>" \
        - For string replacement, the parameters' values should always correspond to what is asked in the prompt, not templates \
        - If there are numbers in a string of the prompt, make sure to preserve them as is \
        - The number of commas generated should be the same as the number of parameters asked, except if there are commas in a parameter of the prompt \
        - The parameters must be in order as the function's definition

    User request: {prompt}

    the parameters are: ("""

    token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
    func_params: list[str] = [f["type"] for f in function["parameters"].values()]
    regex: str = r"-?[\d]+(.[\d]+)?"
    param_nb: int = 1
    param_gen: str = ""
    finished: bool = False

    while not finished:

        logits: list[float] = slm.get_logits_from_input_ids(token_ids)
        sorted_indexes = np.argsort(logits)[::-1]

        for logit_id in sorted_indexes:

            decoded: str = slm.decode([logit_id])

            if ")" in decoded:
                finished = True
                break

            if decoded == "" or decoded == " ":
                continue

            if decoded == ",":
                if param_nb < len(func_params):
                    param_nb += 1
                    param_gen += decoded
                    token_ids.append(logit_id)
                    break

            if func_params[param_nb - 1] == "number" and re.match(regex, (param_gen + decoded)):
                param_gen += decoded
                token_ids.append(logit_id)
                break

            if func_params[param_nb - 1] == "string":
                param_gen += decoded
                token_ids.append(logit_id)
                break

    param_gen = param_gen.strip()
    parameters: list[str] = [p for p in param_gen.split(",") if p]
    ret: list[str | float] = []
    for n in range(len(func_params)):
        if func_params[n] == "number":
            ret.append(float(parameters[n]))
        elif func_params[n] == "string":
            ret.append(parameters[n].strip())
    return ret


def get_function_name(slm: Small_LLM_Model, prompt: str, functions: list[dict[str, str | dict[str, str]]]) -> str:

    llm_prompt: str = f"""
        You are a function calling assistant.
        Available functions: {functions}
        User request: {prompt}
        Call the appropriate function name.
        Function name: """

    token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
    name_generated: str = ""

    while name_generated not in [func["name"] for func in functions]:

        logits: list[float] = slm.get_logits_from_input_ids(token_ids)
        sorted_indexes = np.argsort(logits)[::-1]

        for logit_id in sorted_indexes:

            decoded: str = slm.decode([logit_id])

            if decoded and any([f["name"].startswith(name_generated + decoded) for f in functions]):
                token_ids.append(logit_id)
                name_generated += decoded
                break

    return name_generated


def main() -> None:

    slm: Small_LLM_Model = Small_LLM_Model()
    func_def: str = "data/input/functions_definition.json"
    with open(func_def) as json_file:
        functions = json.load(json_file)
    test_prompts: str = "data/input/function_calling_tests.json"
    with open(test_prompts) as prompt_file:
        prompts = json.load(prompt_file)
    for pr in prompts:
        function_name: str = get_function_name(slm, pr, functions)
        for f in functions:
            if f["name"] == function_name:
                func_info = f
                break
        parameters: list[str | float] = get_parameters(slm, pr, func_info)
        print(f"function name: {function_name} -> parameters: {parameters}")


if __name__ == "__main__":
    main()
