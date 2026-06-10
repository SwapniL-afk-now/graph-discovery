"""Function → OpenAI function-calling JSON schema, from Annotated signatures.

Vendored from DeepVideoDiscovery's dvd/func_call_shema.py (itself adapted from
https://github.com/peterroelants/annotated-docs) so gvd has no dependency on
the DeepVideoDiscovery checkout. Only needs pydantic.

Usage:
    from typing import Annotated as A
    from gvd.func_schema import as_json_schema, doc as D

    def my_tool(x: A[str, D("what x means")]) -> str:
        \"\"\"Tool description for the model.\"\"\"
        ...

    schema = as_json_schema(my_tool)
"""

import inspect
from collections.abc import Callable
from typing import Any, Required, TypedDict

import pydantic
import pydantic.json_schema


class FunctionJSONSchema(TypedDict, total=False):
    name: Required[str]
    description: str
    parameters: dict[str, Any]


def doc(description) -> Any:
    """Annotate a parameter with a description shown to the model."""
    return pydantic.Field(description=description)


def as_json_schema(func: Callable) -> FunctionJSONSchema:
    """Return an OpenAI function-calling JSON schema for ``func``."""
    description = ""
    if func.__doc__:
        description = inspect.cleandoc(func.__doc__).strip()
    return {
        "name": func.__name__,
        "description": description,
        "parameters": _get_parameters_schema(func),
    }


def _get_parameters_schema(func: Callable) -> dict[str, Any]:
    field_definitions: dict[str, tuple[Any, Any]] = {}
    for name, param in inspect.signature(func).parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"`{func.__name__}` parameter `{name!s}` has no annotation; "
                "annotate it to generate the function specification."
            )
        if param.default == inspect.Parameter.empty:
            field_definitions[name] = (param.annotation, pydantic.Field(...))
        else:
            field_definitions[name] = (param.annotation, param.default)
    model = pydantic.create_model("", **field_definitions)  # type: ignore
    return model.model_json_schema(
        schema_generator=_GenerateJsonSchemaNoTitle, mode="validation")


class _GenerateJsonSchemaNoTitle(pydantic.json_schema.GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema.pop("title", None)
        return json_schema

    def get_schema_from_definitions(self, json_ref):
        json_schema = super().get_schema_from_definitions(json_ref)
        if json_schema:
            json_schema.pop("title", None)
        return json_schema

    def field_title_should_be_set(self, schema) -> bool:
        return False
