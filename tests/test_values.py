from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from purrcept_core.models import JsonValue

from purrcept_litellm import LiteLLMConfig
from purrcept_litellm._values import (
    freeze_json_object,
    integer,
    json_object,
    json_value,
    non_empty_string,
    read,
    sequence,
    thaw_json,
)


@dataclass
class ValueObject:
    field: str


class Dumpable:
    def model_dump(self) -> object:
        return {"value": 1}


class BadDumpable:
    def model_dump(self) -> object:
        return object()


def test_sdk_value_reading_and_normalization_helpers() -> None:
    value = ValueObject("value")

    assert read({"field": "mapping"}, "field", None) == "mapping"
    assert read(value, "field", None) == "value"
    assert read(value, "missing", "fallback") == "fallback"
    assert sequence([1, 2]) == (1, 2)
    assert sequence("not-a-sequence") == ()
    assert non_empty_string("value") == "value"
    assert non_empty_string("") is None
    assert non_empty_string(1) is None
    assert integer(2) == 2
    assert integer(True) == 0
    assert integer(-1, default=4) == 4


def test_json_projection_handles_sdk_objects_and_drops_unsupported_values() -> None:
    projected = json_value(
        {
            "finite": 1.5,
            "infinite": float("inf"),
            "none": None,
            "sequence": [1, object(), None],
            "dumped": Dumpable(),
            "bad_dump": BadDumpable(),
            1: "bad-key",
        }
    )

    assert projected == {
        "finite": 1.5,
        "none": None,
        "sequence": [1, None],
        "dumped": {"value": 1},
    }
    assert json_value(object()) is None
    assert json_object(["not", "an", "object"]) == {}


def test_freeze_and_thaw_json_round_trip() -> None:
    frozen = freeze_json_object(
        {"nested": {"items": [1, True, None]}},
        field_name="fixture",
    )

    assert frozen["nested"] == {"items": (1, True, None)}
    assert thaw_json(cast(JsonValue, frozen)) == {"nested": {"items": [1, True, None]}}


def test_sequence_reference_cycles_are_rejected() -> None:
    values: list[JsonValue] = []
    values.append(cast(JsonValue, values))

    with pytest.raises(ValueError, match="reference cycle"):
        LiteLLMConfig(default_options={"values": values})
