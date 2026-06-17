import os
import json
from .llm_sdk import Small_LLM_Model
from src import Function, Parser, Constraint


def main() -> None:

    """

    Run the function-calling pipeline end to end.

    Loads functions and prompts, generates a function name and
    parameters for each prompt via the SLM, then writes the
    results to the output JSON file.

    """

    try:

        paths = Parser.check_args()

        functions: list[Function]
        prompts: list[dict[str, str]]

        functions, prompts = Parser.verify_files(
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
