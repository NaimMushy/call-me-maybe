from llm_sdk import Small_LLM_Model
import sys


def main() -> None:

    slm: Small_LLM_Model = Small_LLM_Model()
    decode_prompt(slm, sys.argv[1])


def decode_prompt(slm: Small_LLM_Model, prompt: str) -> None:

    prompt_ids: list[int] = slm.encode(prompt).tolist()
    logits: list[float] = slm.get_logits_from_input_ids(prompt_ids)
    for logit in range(20):
        print(f"token number {logit} : {slm.decode([prompt_ids[logit]])} -> probability : {logits[logit]}\n")
        # look if the token associated will maintain structure and schema
        # if not, remove it from the logits
        # else keep it
    # choose token that has the logit with best probability


if __name__ == "__main__":
    main()
