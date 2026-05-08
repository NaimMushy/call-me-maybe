import numpy as np
import json
from .llm_sdk import Small_LLM_Model


def get_function_name(slm: Small_LLM_Model, prompt: str, functions: list[dict[str, str | dict[str, str]]]) -> str:

    llm_prompt: str = (
            "You are a function calling assistant.\n" +
            f"You have these functions at your disposal : {functions}.\n" +
            f"Here is the user request: {prompt}.\n"
            "Generate the function name corresponding to the request."
    )

    token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
    name_generated: str = ""

    while name_generated not in [func["name"] for func in functions]:

        logits: list[float] = slm.get_logits_from_input_ids(token_ids)
        sorted_indexes = np.argsort(logits)

        for logit_id in sorted_indexes:

            decoded: str = slm.decode([logit_id])

            if decoded and any([f["name"].startswith(name_generated + decoded) for f in functions]):
                token_ids.append(logit_id)
                name_generated += decoded
                break

    return name_generated


def main() -> None:

    func_def: str = "data/input/functions_definition.json"
    with open(func_def) as json_file:
        functions = json.load(json_file)
    slm: Small_LLM_Model = Small_LLM_Model()
    prompt = "Add the numbers 2 and 3"
    print(get_function_name(slm, prompt, functions))


if __name__ == "__main__":
    main()
