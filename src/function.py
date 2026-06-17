from pydantic import BaseModel, model_validator
from typing_extensions import Self as self


class Function(BaseModel):

    """
    A class used to store a function's attributes and definition.

    Attributes
    ----------
    name: str
        Name of the function.
    description: str
        Short description of the function's purpose.
    parameters: dict[str, dict[str, str]]
        Mapping of parameter names to their type definition.
    returns: dict[str, str]
        Mapping describing the return type of the function.
    """

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validate_parameters(self) -> self:

        """

        Validate the structure and types of the attribute 'parameters'.

        Returns
        -------
        self
            The validated model instance.

        Raises
        ------
        ValueError
            If a parameter key, value, or type is invalid.

        """

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

        """

        Validate the structure and types of the attribute 'returns'.

        Returns
        -------
        self
            The validated model instance.

        Raises
        ------
        ValueError
            If a return key or value is invalid.

        """

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
