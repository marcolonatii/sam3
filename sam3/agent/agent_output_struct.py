from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model


#todo: codex's code
PRIMITIVE_TYPE_MAP: Dict[str, type] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "dict": dict,
    "object": dict,
    "list": list,
    "array": list,
}


def _normalize_field_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name or "field"


def _type_from_token(token: Any) -> Optional[type]:
    if isinstance(token, str):
        token = token.strip().lower()
        return PRIMITIVE_TYPE_MAP.get(token)
    if token in (int, float, str, bool, list, dict):
        return token
    return None


def _build_field_schema(
    value: Any, field_name: str, model_name: str
) -> Tuple[type, Field]:
    if isinstance(value, dict):
        nested_model = FormatSchemaBuilder.from_format_dict(
            value, model_name=f"{model_name}{_normalize_field_name(field_name).title()}"
        )
        return nested_model, Field(..., description=field_name)

    if isinstance(value, list):
        item_schema: Any = str
        for item in value:
            if isinstance(item, str) and item.strip() == "...":
                continue
            if item is None:
                continue
            item_schema = item
            break
        item_type = _type_from_token(item_schema)
        if item_type is not None:
            return List[item_type], Field(..., description=field_name)
        if isinstance(item_schema, dict):
            nested_model = FormatSchemaBuilder.from_format_dict(
                item_schema, model_name=f"{model_name}{_normalize_field_name(field_name).title()}Item"
            )
            return List[nested_model], Field(..., description=field_name)
        return List[str], Field(..., description=field_name)

    primitive_type = _type_from_token(value)
    if primitive_type is not None:
        return primitive_type, Field(..., description=field_name)

    return str, Field(..., description=field_name)


class FormatSchemaBuilder:
    @staticmethod
    def from_format_dict(format_dict: Dict[str, Any], model_name: str = "TaskOutput") -> Type[BaseModel]:
        fields: Dict[str, Tuple[type, Field]] = {}
        for key, value in format_dict.items():
            if isinstance(value, str) and value.strip() == "...":
                continue
            field_name = _normalize_field_name(key)
            fields[field_name] = _build_field_schema(value, key, model_name)

        return create_model(model_name, **fields)

    @classmethod
    def from_json_file(cls, json_path: str, model_name: Optional[str] = None) -> Type[BaseModel]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Format JSON must be an object at the top level.")

        if model_name is None:
            model_name = "TaskOutput"
        return cls.from_format_dict(data, model_name=model_name)

if __name__ == "__main__":
    data = {
        "score_list": [
            {
                "frame_idx": int,
                "score": int
            }
        ]
    }
    model = FormatSchemaBuilder.from_format_dict(data)
    print(model.model_json_schema())