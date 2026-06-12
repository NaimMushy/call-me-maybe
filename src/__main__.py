import numpy as np
import json
import regex
import os
import sys
from .llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field, model_validator


class Function(BaseModel):

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters(self) -> None:

        valid_types: list[str] = [
            "int", "number", "num", "integer", "float",
            "boolean", "bool", "str", "string"
        ]

        for param_name, param_value in self.parameters.items():

            if not isinstance(param_name, str):

                raise ValueError(
                    f"Invalid key '{param_name}' "
                    f"for parameter of function '{self.name}'"
                )

            if not isinstance(param_value, dict):

                raise ValueError(
                    f"Invalid parameter '{param_name}' "
                    f"for function '{self.name}' - Should be a dictionary"
                )

            keys: list[str] = [k for k in param_value.keys()]

            if len(keys) > 1:

                raise ValueError(
                    f"Too many types provided for parameter '{param_name}' "
                    f"of function '{self.name}'"
                )

            if keys[0] != "type":

                raise ValueError(
                    f"Invalid key for parameter '{param_name}' "
                    f"of function '{self.name}'"
                )

            if (
                not isinstance(param_value[keys[0]], str)
                or param_value[keys[0]] not in valid_types
            ):

                raise ValueError(
                    f"Invalid type '{param_value[keys[0]]}' "
                    f"for parameter '{param_name}' of function '{self.name}'"
                )

    @model_validator(mode="after")
    def validate_return_types(self) -> None:

        valid_types: list[str] = [
            "int", "number", "num", "integer", "float",
            "boolean", "bool", "str", "string"
        ]

        for param_name, param_value in self.returns.items():

            if not isinstance(param_value, str):

                raise ValueError(
                    f"Invalid key '{param_name}' "
                    f"for return value of function '{self.name}' "
                    "- Should be a string"
                )

            if not isinstance(param_value, str):

                raise ValueError(
                    f"Invalid return value '{param_name}' "
                    f"for function '{self.name}' - Should be a string"
                )

            if param_name != "type":

                raise ValueError(
                    f"Invalid key '{param_name}' for return value "
                    f"of function '{self.name}' - Should be 'type'"
                )

            if param_value not in valid_types:

                raise ValueError(
                    f"Invalid type '{param_value}' "
                    f"for return value of function '{self.name}'"
                )


def check_args(args: list[str]) -> dict[str, str]:

    if len(args) > 6:
        raise ValueError("Too many arguments!")

    files_dict: dict[str, str] = {}

    for arg in range(len(args)):

        if arg % 2 == 1:

            continue

        if not (match := regex.match(
            r"--(functions_definition|input|output)",
            args[arg]
        )):

            raise ValueError(f"Invalid flag '{args[arg]}'!")

        if match.group(1) in files_dict.keys():

            raise ValueError(f"Flag '{match.group(1)}' is already defined!")

        if arg == len(args) - 1 or args[arg + 1].startswith("--"):

            raise ValueError(
                f"No associated value given for flag '{args[arg]}'!"
            )

        if len(args[arg + 1].split("/")) < 1:

            raise ValueError(f"Invalid path given for flag '{args[arg]}'")

        files_dict[match.group(1)] = args[arg + 1]

    if "input" not in files_dict.keys():

        files_dict["input"] = "data/input/function_calling_tests.json"

    if "output" not in files_dict.keys():

        files_dict["output"] = "data/output/function_calling_results.json"

    if "functions_definition" not in files_dict.keys():

        files_dict["functions_definition"] = "data/input/functions_definition.json"

    return files_dict


def verify_files(
    fc_def_path: str,
    pr_path: str
) -> [list[Function], list[dict[str, str]]]:

    with open(fc_def_path) as func_file:

        loaded_functions = json.load(func_file)

    if not loaded_functions:

        raise ValueError("No function definitions in input file")

    functions: list[Function] = []

    mandatory: list[str] = ["name", "description", "parameters", "returns"]

    for func in loaded_functions:

        func_params: set = set()

        for param, param_value in func.items():

            if param in mandatory:

                if param in func_params:

                    raise ValueError(
                        f"Parameter '{param}' has already been defined!"
                    )

                func_params.add(param)

            else:

                raise ValueError(
                    f"Invalid parameter '{param}' in function definition"
                )

        if any([m not in func_params for m in mandatory]):

            raise ValueError(
                f"Missing parameters for function definition '{func}'!"
            )

        functions.append(Function(
            name=func["name"],
            description=func["description"],
            parameters=func["parameters"],
            returns=func["returns"]
        ))

    with open(pr_path) as prompts_file:

        test_prompts = json.load(prompts_file)

    for pr in test_prompts:

        if len([k for k in pr.keys()]) > 1:

            raise ValueError(
                f"Invalid prompt '{pr}' "
                f"in function calling file - Too many keys"
            )

        if "prompt" not in pr.keys():

            raise ValueError(
                f"Invalid key in prompt dict '{pr}' - Should be 'prompt'"
            )

    return functions, test_prompts


class Constraint(BaseModel):

    functions: list[Function]
    regexes: dict[str, str] = Field(default={
        "number": r"[-]?[\d]+([.][\d]+)?",
        "boolean": r"true|false",
        "string": r'[^"]*'
    })
    types: dict[str, str] = Field(default={
        "string": "string",
        "str": "string",
        "number": "number",
        "num": "number",
        "int": "number",
        "integer": "number",
        "float": "number",
        "bool": "boolean",
        "boolean": "boolean"
    })

    def get_single_parameter(
        self,
        slm: Small_LLM_Model,
        llm_prompt: str,
        pattern: str,
        end_char: str
    ) -> str:

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

    def get_parameters(
        self,
        slm: Small_LLM_Model,
        prompt: dict[str, str],
        function_name: str
    ) -> dict[str, str | float | bool]:

        for f_id in self.functions:

            if f_id.name == function_name:
                function: Function = f_id

        llm_prompt = f"""You are a function calling assistant. \
        Your job is to return the parameters of the function in the \
        prompt. \

        function: \
            name: {function.name} \
            description: {function.description} \
            parameters: {function.parameters} \

        The parameters must have the same type as in the function. \
        Numbers or boolean parameters \
        must be inside parentheses, example : "(42)". \
        Strings must be inside double quotes, example : "hello". \

        Examples: \
            - Function: fn_substitute_string_with_regex \
            - User request: replace all digits with NUM \
            - Parameters: (source_string: "hello 42", \
            regex: "([\\d]+)", replacement: "NUM") \

            - Function: fn_substitute_string_with_regex \
            - User request: replace all vowels with an asterisk symbol \
            - Parameters: (source_string: "hello world", \
            regex: "([aeiou])", replacement: "*") \

        Do not include the user request in your answer. \

        Here is your user request: {prompt["prompt"]}. \

        Parameters: ("""

        parameters: dict[str, str | float | bool] = {}
        types: list[str] = [f["type"] for f in function.parameters.values()]

        for p_id, p_type in enumerate(types):

            param: str = ""
            param_name: str = ([k for k in function.parameters.keys()][p_id])
            # print(f"parameter name: {param_name}")
            llm_prompt += param_name + ": "

            if self.types[p_type] == "string":

                llm_prompt += '"'
                param = '"'
                content: str = self.get_single_parameter(
                    slm,
                    llm_prompt,
                    self.regexes[self.types[p_type]],
                    '"'
                )

                if len(content) > 0 and content[0] == " ":

                    if len(content) > 1 and content[1] != " ":
                        content = content[1:]

                param += content + '"'
                param = param.strip('"').replace("\\'", "'")

            else:

                param = self.get_single_parameter(
                    slm,
                    llm_prompt,
                    self.regexes[self.types[p_type]],
                    ")"
                )

            # print(f"parameter: {param}")
            llm_prompt += param

            if p_id == len(types) - 1:

                llm_prompt += ")"

            else:

                llm_prompt += ", "

            if self.types[p_type] == "bool":

                parameters[param_name] = bool(param)

            elif self.types[p_type] == "number":

                parameters[param_name] = float(param)

            else:

                parameters[param_name] = param

        return parameters

    def get_function_name(self, slm: Small_LLM_Model, prompt: dict) -> str:

        llm_prompt: str = f"""
            You are a function calling assistant.
            Available functions: {self.functions}
            User request: {prompt}
            Call the appropriate function name.
            Function name: """

        token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
        name_generated: str = ""

        while name_generated not in [func.name for func in self.functions]:

            logits: list[float] = slm.get_logits_from_input_ids(token_ids)
            sorted_indexes = np.argsort(logits)[::-1]

            for logit_id in sorted_indexes:

                decoded: str = slm.decode([logit_id])

                if decoded and any([
                    f.name.startswith(name_generated + decoded)
                    for f in self.functions
                ]):

                    token_ids.append(logit_id)
                    name_generated += decoded
                    break

        return name_generated


def main() -> None:

    try:

        paths: dict[str, str] = check_args(sys.argv[1:])

        functions: list[Function]
        prompts: list[dict[str, str]]

        functions, prompts = verify_files(
            paths["functions_definition"],
            paths["input"]
        )

    except Exception as err:

        print(f"Caught {err.__class__.__name__}: {err}")
        return None

    constr: Constraint = Constraint(functions=functions)
    slm: Small_LLM_Model = Small_LLM_Model()
    dicts: list[dict[str, str | dict]] = []

    for pr in prompts:

        json_obj: dict[str, str | dict[str, str]] = {"prompt": pr["prompt"]}
        json_obj["name"] = constr.get_function_name(slm, pr)
        json_obj["parameters"] = constr.get_parameters(
            slm,
            pr,
            json_obj["name"]
        )

        dicts.append(json_obj)

        print(
            f"function name: {json_obj['name']} "
            f"-> parameters: {json_obj['parameters']}\n"
        )

    if len(paths["output"].split("/")) > 1:

        os.makedirs("/".join(paths["output"].split("/")[:-1]), exist_ok=True)

    with open(paths["output"], "w") as output_file:

        output_file.write(json.dumps(dicts, indent=4))


if __name__ == "__main__":
    main()
