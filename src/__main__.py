import numpy as np
import json
import regex
import time
import os
from .llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field


class Constraint(BaseModel):

    functions: list[dict[str, str | dict[str, str | dict[str, str]]]]
    regexes: dict[str, str] = Field(default={
        "number": r"[-]?[\d]+([.][\d]+)?",
        "boolean": r"true|false",
        "string": r'[^"]*'
    })

    def get_single_parameter(self, slm: Small_LLM_Model, llm_prompt: str, pattern: str, end_char: str) -> str:

        token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
        gen: str = ""
        finished: bool = False
        max_chars: int = 200

        while not finished:

            if len(gen) > max_chars:
                break

            logits: list[float] = slm.get_logits_from_input_ids(token_ids)
            sorted_indexes = np.argsort(logits)[::-1]

            if len(gen) == 0:
                for i in sorted_indexes:
                    decoded: str = slm.decode([i])
                    if decoded.startswith(" ") and decoded.strip().isalpha():
                        logits[i] = -float("inf")
                sorted_indexes = np.argsort(logits)[::-1]

            for logit_id in sorted_indexes:

                decoded = slm.decode([logit_id])
                # print(f"decoded : {decoded}")

                is_complete = regex.fullmatch(pattern, gen) is not None
                if is_complete and end_char in decoded.strip():
                    ending: str = ""
                    for c in decoded.strip():
                        if c == end_char:
                            break
                        ending += c
                    gen += ending
                    finished = True
                    break

                if regex.fullmatch(pattern, gen + decoded, partial=True):

                    gen += decoded
                    token_ids.append(logit_id)
                    break

                # time.sleep(0.5)

        return gen

    def get_parameters(self, slm: Small_LLM_Model, prompt: str, function_name: str) -> dict[str, str | float | bool]:

        for f_id in self.functions:

            if f_id["name"] == function_name:
                function: dict[str, str | dict[str, str]] = f_id

        llm_prompt = f"""You are a function calling assistant. \
        Your job is to return the parameters of the function in the \
        prompt.

        function:
            name: {function['name']}
            description: {function['description']}
            parameters: {function['parameters']}

        The parameters must have the same type as in the function. \
        Numbers or boolean parameters must be inside parentheses, example : "(42)". \
        Strings must be inside double quotes, example : "hello". \

        Examples:
            - Function: fn_substitute_string_with_regex
            - User request: replace all digits with NUM
            - Parameters: (source_string: "hello 42", regex: "([\\d]+)", replacement: "NUM")

            - Function: fn_substitute_string_with_regex
            - User request: replace all vowels with an asterisk symbol
            - Parameters: (source_string: "hello world", regex: "([aeiou])", replacement: "*")

        User request: {prompt}.

        Parameters: ("""

        parameters: dict[str, str | float | bool] = {}
        types: list[str] = [f["type"] for f in function["parameters"].values()]

        for p_id, p_type in enumerate(types):

            param: str = ""
            param_name: str = ([k for k in function["parameters"].keys()][p_id])
            # print(f"parameter name: {param_name}")
            llm_prompt += param_name + ": "
            if p_type == "string":
                llm_prompt += '"'
                param = '"'
                content: str = self.get_single_parameter(slm, llm_prompt, self.regexes[p_type], '"')

                if len(content) > 0 and content[0] == " ":
                    if len(content) > 1 and content[1] != " ":
                        content = content[1:]

                param += content + '"'
                param = param.strip('"').replace("\\'", "'")
            else:
                param = self.get_single_parameter(slm, llm_prompt, self.regexes[p_type], ")")
            # print(f"parameter: {param}")
            llm_prompt += param
            if p_id == len(types) - 1:
                llm_prompt += ")"
            else:
                llm_prompt += ", "
            if p_type == "bool":
                parameters[param_name] = bool(param)
            elif p_type == "number":
                parameters[param_name] = float(param)
            else:
                parameters[param_name] = param

        return parameters

    def get_function_name(self, slm: Small_LLM_Model, prompt: str) -> str:

        llm_prompt: str = f"""
            You are a function calling assistant.
            Available functions: {self.functions}
            User request: {prompt}
            Call the appropriate function name.
            Function name: """

        token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
        name_generated: str = ""

        while name_generated not in [func["name"] for func in self.functions]:

            logits: list[float] = slm.get_logits_from_input_ids(token_ids)
            sorted_indexes = np.argsort(logits)[::-1]

            for logit_id in sorted_indexes:

                decoded: str = slm.decode([logit_id])

                if decoded and any([f["name"].startswith(name_generated + decoded) for f in self.functions]):
                    token_ids.append(logit_id)
                    name_generated += decoded
                    break

        return name_generated


def main() -> None:

    func_def: str = "data/input/functions_definition.json"
    with open(func_def) as json_file:
        functions = json.load(json_file)
    slm: Small_LLM_Model = Small_LLM_Model()
    constr: Constraint = Constraint(
        functions=functions
    )
    test_prompts: str = "data/input/function_calling_tests.json"
    with open(test_prompts) as prompt_file:
        prompts = json.load(prompt_file)
    dicts: list[dict[str, str | float | bool]] = []
    for pr in prompts:
        function_name: str = constr.get_function_name(slm, pr)
        parameters: dict[str, str | float | bool] = constr.get_parameters(slm, pr, function_name)
        dicts.append(parameters)
        print(f"function name: {function_name} -> parameters: {parameters}\n")
    os.makedirs("data/output", exist_ok=True)
    with open("data/output/function_calling_results.json", "w") as output_file:
        output_file.write(json.dumps(dicts, indent=4))


if __name__ == "__main__":
    main()
