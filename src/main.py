import numpy as np
import json
from .llm_sdk import Small_LLM_Model


def get_function_name(slm: Small_LLM_Model, prompt: str, functions: list[dict[str, str | dict[str, str]]]) -> str:

    llm_prompt: str = (
            "You are a function calling assistant.\n" +
            f"You have these functions at your disposal : {json.dumps(functions)}.\n" +
            f"Here is the user request: {prompt}.\n"
            "Generate the appropriate function name."
    )

    token_ids: list[int] = slm.encode(llm_prompt)[0].tolist()
    name_generated: str = ""

    while name_generated not in [func["name"] for func in functions]:

        logits: list[float] = slm.get_logits_from_input_ids(token_ids)
        sorted_indexes = np.argsort(logits)

        for logit_id in sorted_indexes:

            decoded: str = slm.decode([logits[logit_id]])

            if any([f["name"].startswith(name_generated + decoded) for f in functions]):
                token_ids.append(logit_id)
                name_generated += decoded
                break

    return name_generated
