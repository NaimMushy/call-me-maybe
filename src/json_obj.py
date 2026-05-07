from pydantic import BaseModel


class JSON_Container(BaseModel):

    prompts: list[str] = []
    func_names: list[str] = []
    parameters: list[dict[str, str | int | float]] = []
