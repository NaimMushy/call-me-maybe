from pydantic import BaseModel, Field
import numpy as np
import regex
import textwrap
from .llm_sdk import Small_LLM_Model
from src.function import Function


class Constraint(BaseModel):

    """

    A class to apply constraint decoding
    to generate proper responses using a SLM.

    Attributes
    ----------
    functions: list[Function]
        Available function definitions to call.
    regexes: dict[str, str]
        Regex patterns used to validate generated values
        per type.
    types: dict[str, str]
        Mapping from declared parameter types to internal
        type categories.

    """

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

        """

        Generate one parameter value token by token via a SLM.

        Parameters
        ----------
        slm: Small_LLM_Model
            The small language model used for generation.
        llm_prompt: str
            The prompt provided to the model so far.
        pattern: str
            Regex pattern the generated value must match.
        end_char: str
            Character marking the end of the value.

        Returns
        -------
        str
            The generated parameter value as a string.

        """

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

        """

        Generate all parameters for a given function call.

        Parameters
        ----------
        slm: Small_LLM_Model
            The small language model used for generation.
        prompt: dict[str, str]
            The user prompt, with key "prompt".
        function_name: str
            Name of the function to generate parameters for.

        Returns
        -------
        dict[str, str | bool | int | float]
            Mapping of parameter names to generated values.

        """

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
        In regexes, "spaces"=" ", "asterisks"="*" and so on.
        Separate arguments by commas.

        Examples:
        - User request: calculate the square root of forty-two.
        - Parameters: (a: 42)

        - User request: replace all vowels in '' with asterisks
        - Parameters: (source_string: "", \
regex: "([aeiouAEIOU])", replacement: "*")

        - User request: replace all digits in 'hello 42' with NUM
        - Parameters: (source_string: "hello 42", \
regex: "([0-9]+)", replacement: "NUM")

        Do not include the user request in your answer.

        Here is your user request: {prompt["prompt"]}.

        Parameters: (""")

        parameters: dict[str, str | float | bool | int] = {}
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

        """

        Determine which function the user prompt refers to.

        Parameters
        ----------
        slm: Small_LLM_Model
            The small language model used for generation.
        prompt: dict[str, str]
            The user prompt, with key "prompt".

        Returns
        -------
        str
            The name of the matched function.

        """

        llm_prompt: str = textwrap.dedent(f"""
            You are a function calling assistant.
            Available functions: {self.functions}
            User request: {prompt}
            Call the appropriate function name.
            Function name: """)

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
