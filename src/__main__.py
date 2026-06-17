import numpy as np
import json
import regex
import argparse
import os
import textwrap
from .llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field, model_validator
from typing import Any
from typing_extensions import Self as self


class Function(BaseModel):

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters(self) -> self:

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

        return self

    @model_validator(mode="after")
    def validate_return_types(self) -> self:

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

        return self


def check_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="the path to the prompt input file"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="the path to the functions' definition file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="the path to the output file"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="toggle the verbose output log"
    )
    args: argparse.Namespace = parser.parse_args()
    return args


def open_json(
    path: str
) -> Any:

    try:
        with open(path) as file:

            loaded_json = json.load(file)

    except json.JSONDecodeError as jerr:

        raise ValueError(f"Invalid JSON file: {jerr}")

    except OSError as oserr:

        raise ValueError(
            f"Impossible to open the json file '{path}' : {oserr}"
        )

    except UnicodeDecodeError as uderr:

        raise ValueError(
            f"Non utf-8 characters in the json file '{path}' : {uderr}"
        )

    return loaded_json


def verify_files(
    fc_def_path: str,
    pr_path: str
) -> tuple[list[Function], list[dict[str, str]]]:

    loaded_functions: list[dict[str, Any]] = open_json(fc_def_path)

    if not loaded_functions:

        raise ValueError("No function definitions in input file")

    functions: list[Function] = []

    mandatory: list[str] = ["name", "description", "parameters", "returns"]

    for func in loaded_functions:

        func_params: set[str] = set()

        for param, param_value in func.items():

            if not param_value:

                raise ValueError(
                    f"The value of the parameter '{param}' "
                    "has not been defined"
                )

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

    test_prompts: list[dict[str, str]] = open_json(pr_path)

    if not test_prompts:

        raise ValueError(f"No prompts given in input file '{pr_path}'")

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

        if not isinstance(pr["prompt"], str):

            raise ValueError(
                f"Invalid prompt '{pr['prompt']}' - Should be a string"
            )

        if not pr["prompt"]:

            raise ValueError(
                f"Invalid prompt '{pr['prompt']}' "
                "- Should not be an empty string"
            )

    return functions, test_prompts


class Constraint(BaseModel):

    functions: list[Function]
    regexes: dict[str, str] = Field(default={
        "number": r"[-]?[\d]+([.][\d]+)?",
        "boolean": r"true|false",
        "string": r'([^"\\]|\\.){0,1000}'
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

            for logit_id in sorted_indexes:

                decoded = slm.decode([logit_id])

                if (
                    regex.fullmatch(pattern, gen)
                    and end_char in decoded.strip()
                ):

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

        llm_prompt = textwrap.dedent(f"""
        Generate the parameters of the function in the prompt.

        function name: {function.name}
        function description: {function.description}
        function parameters: {function.parameters}

        Strings must be inside double quotes, example : "hello".
        Keep empty strings and literal space/' ' in user request.
        Separate arguments by commas.

        Examples:
        - User request: calculate the square root of forty-two.
        - Parameters: (a: 42)

        - User request: replace all digits in 'hello 42' with NUM
        - Parameters: (source_string: "hello 42", \
regex: "([0-9]+)", replacement: "NUM")

        - User request: replace all vowels in '' with spaces
        - Parameters: (source_string: "", \
regex: "([aeiouAEIOU])", replacement: "' '")

        Do not include the user request in your answer.

        Here is your user request: {prompt["prompt"]}.

        Parameters: (""")

        parameters: dict[str, str | float | bool] = {}
        types: list[str] = [f["type"] for f in function.parameters.values()]
        end_char: str = ("," if len(types) > 1 else ")")

        for p_id, p_type in enumerate(types):

            param: str = ""
            param_name: str = ([k for k in function.parameters.keys()][p_id])
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
                param = param.strip('"')

            else:

                param = self.get_single_parameter(
                    slm,
                    llm_prompt,
                    self.regexes[self.types[p_type]],
                    end_char
                )

            llm_prompt += param

            if p_id == len(types) - 1:

                end_char = ")"
                llm_prompt += end_char

            else:

                end_char = ","
                llm_prompt += end_char + " "

            if self.types[p_type] == "bool":

                parameters[param_name] = bool(param)

            elif p_type in ["number", "num", "float"]:

                parameters[param_name] = float(param)

            elif p_type in ["integer", "int"]:

                parameters[param_name] = int(param)

            else:

                parameters[param_name] = param

        return parameters

    def get_function_name(
        self,
        slm: Small_LLM_Model,
        prompt: dict[str, str]
    ) -> str:

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

        paths = check_args()

        functions: list[Function]
        prompts: list[dict[str, str]]

        functions, prompts = verify_files(
            paths.functions_definition,
            paths.input
        )

    except Exception as err:

        print(f"Caught {err.__class__.__name__}: {err}")
        return None

    constr: Constraint = Constraint(functions=functions)
    slm: Small_LLM_Model = Small_LLM_Model()
    dicts: list[
        dict[str, str | dict[str, str | float | bool]]
    ] = []

    for pr in prompts:

        json_obj: dict[str, str | dict[str, str | float | bool]] = {
            "prompt": pr["prompt"]
        }
        json_obj["name"] = constr.get_function_name(slm, pr)
        json_obj["parameters"] = constr.get_parameters(
            slm,
            pr,
            str(json_obj["name"])
        )

        dicts.append(json_obj)

        if paths.verbose:
            print(
                f"function name: {json_obj['name']} "
                f"-> parameters: {json_obj['parameters']}\n"
            )

    if len(paths.output.split("/")) > 1:

        os.makedirs("/".join(paths.output.split("/")[:-1]), exist_ok=True)

    try:

        with open(paths.output, "w") as output_file:

            output_file.write(json.dumps(dicts, indent=4))

    except OSError as oserr:

        print(f"Caught OSError for output file '{paths.output}' : {oserr}")

    except Exception as err:

        print(f"Caught {err.__class__.__name__}: {err}")


if __name__ == "__main__":
    main()
