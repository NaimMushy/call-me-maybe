import argparse
import json
from pydantic import BaseModel
from typing import Any
from src.function import Function


class Parser(BaseModel):

    """

    A helper class to parse program arguments and input files.

    """

    @staticmethod
    def check_args() -> argparse.Namespace:

        """

        Parse command-line arguments for the script.

        Returns
        -------
        argparse.Namespace
            Parsed arguments: input, functions_definition, output,
            and verbose.

        """

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

    @staticmethod
    def open_json(
        path: str
    ) -> Any:

        """

        Load and parse a JSON file.

        Parameters
        ----------
        path: str
            Path to the JSON file to load.

        Returns
        -------
        Any
            The parsed JSON content.

        Raises
        ------
        ValueError
            If the file is invalid JSON, cannot be opened, or
            contains non-UTF-8 characters.

        """

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

    @staticmethod
    def verify_files(
        fc_def_path: str,
        pr_path: str
    ) -> tuple[list[Function], list[dict[str, str]]]:

        """

        Load and validate function definitions and test prompts.

        Parameters
        ----------
        fc_def_path: str
            Path to the function definitions JSON file.
        pr_path: str
            Path to the test prompts JSON file.

        Returns
        -------
        tuple[list[Function], list[dict[str, str]]]
            The validated functions and the validated prompts.

        Raises
        ------
        ValueError
            If a function definition or prompt is missing,
            malformed, or contains invalid keys/values.

        """

        loaded_functions: list[dict[str, Any]] = Parser.open_json(fc_def_path)

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

        test_prompts: list[dict[str, str]] = Parser.open_json(pr_path)

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
