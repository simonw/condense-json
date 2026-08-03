"""Structural (whole-subtree) replacement values: dicts and lists."""

from typing import Any

import pytest

from condense_json import UncondenseError, condense_json, uncondense_json

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Full name of the person"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


def test_whole_dict_subtree_is_replaced() -> None:
    obj: dict[str, Any] = {"text": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    condensed = condense_json(obj, {"s": SCHEMA})
    assert condensed == {
        "text": {"format": {"type": "json_schema", "schema": {"$": "s"}}}
    }
    assert uncondense_json(condensed, {"s": SCHEMA}) == obj


def test_key_order_does_not_matter() -> None:
    reordered: dict[str, Any] = {
        "required": ["name", "age"],
        "properties": {
            "age": {"type": "integer"},
            "name": {"type": "string", "description": "Full name of the person"},
        },
        "type": "object",
    }
    condensed = condense_json({"schema": reordered}, {"s": SCHEMA})
    assert condensed == {"schema": {"$": "s"}}
    # Resolution substitutes the replacement's form, structurally equal
    # to what was condensed
    assert uncondense_json(condensed, {"s": SCHEMA}) == {"schema": SCHEMA}


def test_list_replacement() -> None:
    steps: list[Any] = ["wash", "rinse", {"repeat": True}]
    obj: dict[str, Any] = {"a": ["wash", "rinse", {"repeat": True}], "b": "unrelated"}
    condensed = condense_json(obj, {"steps": steps})
    assert condensed == {"a": {"$": "steps"}, "b": "unrelated"}
    assert uncondense_json(condensed, {"steps": steps}) == obj


def test_outermost_match_wins() -> None:
    inner = {"type": "integer"}
    outer = {"age": {"type": "integer"}}
    obj = {"properties": {"age": {"type": "integer"}}}
    condensed = condense_json(obj, {"outer": outer, "inner": inner})
    assert condensed == {"properties": {"$": "outer"}}


def test_inner_matches_still_found_when_outer_does_not_match() -> None:
    inner = {"type": "integer"}
    obj: dict[str, Any] = {"properties": {"age": {"type": "integer"}, "extra": True}}
    condensed = condense_json(obj, {"inner": inner})
    assert condensed == {"properties": {"age": {"$": "inner"}, "extra": True}}


def test_string_form_of_a_container_is_not_matched() -> None:
    # A string that happens to contain the JSON serialization of the
    # value must survive untouched - a reference inside a string context
    # must resolve to a string.
    obj: dict[str, Any] = {"as_string": '{"repeat":true}', "as_value": {"repeat": True}}
    replacements: dict[str, Any] = {"r": {"repeat": True}}
    condensed = condense_json(obj, replacements)
    assert condensed == {"as_string": '{"repeat":true}', "as_value": {"$": "r"}}
    assert uncondense_json(condensed, replacements) == obj


def test_first_id_wins_for_duplicate_values() -> None:
    obj = {"a": {"x": 1}}
    condensed = condense_json(obj, {"first": {"x": 1}, "second": {"x": 1}})
    assert condensed == {"a": {"$": "first"}}


def test_scalars_and_empties_are_ignored() -> None:
    obj: dict[str, Any] = {"n": 42, "flag": True, "nothing": None, "d": {}, "l": []}
    replacements: dict[str, Any] = {"a": 42, "b": True, "c": None, "d": {}, "e": []}
    assert condense_json(obj, replacements) == obj


def test_marker_shaped_subtree_matching_a_replacement_round_trips() -> None:
    weird = {"$": "i-am-data"}
    obj = {"payload": {"$": "i-am-data"}}
    replacements = {"w": weird}
    condensed = condense_json(obj, replacements)
    assert condensed == {"payload": {"$": "w"}}
    assert uncondense_json(condensed, replacements) == obj


def test_marker_shaped_subtree_not_matching_is_still_escaped() -> None:
    obj = {"payload": {"$": "i-am-data"}}
    condensed = condense_json(obj, {"unrelated": {"x": 1}})
    assert condensed == {"payload": {"$raw": {"$": "i-am-data"}}}
    assert uncondense_json(condensed, {"unrelated": {"x": 1}}) == obj


def test_structural_and_substring_replacements_compose() -> None:
    novel = "It was the best of times, it was the worst of times"
    obj: dict[str, Any] = {
        "quote": f"Opening: {novel}",
        "schema": SCHEMA,
    }
    replacements = {"n": novel, "s": SCHEMA}
    condensed = condense_json(obj, replacements)
    assert condensed == {
        "quote": {"$r": ["Opening: ", {"$": "n"}]},
        "schema": {"$": "s"},
    }
    assert uncondense_json(condensed, replacements) == obj


def test_uncondensed_containers_are_independent_copies() -> None:
    replacements = {"s": {"x": [1, 2]}}
    condensed = {"a": {"$": "s"}, "b": {"$": "s"}}
    result = uncondense_json(condensed, replacements)
    result["a"]["x"].append(3)
    # Sibling markers and the replacements mapping are unaffected
    assert result["b"] == {"x": [1, 2]}
    assert replacements["s"] == {"x": [1, 2]}


def test_container_id_inside_r_segments_raises() -> None:
    condensed: dict[str, Any] = {"text": {"$r": ["prefix ", {"$": "s"}]}}
    with pytest.raises(UncondenseError):
        uncondense_json(condensed, {"s": {"x": 1}})


def test_unknown_id_still_raises() -> None:
    with pytest.raises(UncondenseError):
        uncondense_json({"a": {"$": "missing"}}, {"s": {"x": 1}})


def test_numbers_are_not_conflated_by_canonical_form() -> None:
    # 1 and 1.0 and True serialize differently in canonical form, so a
    # replacement containing one never matches a subtree containing
    # another
    assert condense_json({"a": [1.0]}, {"r": [1]}) == {"a": [1.0]}
    assert condense_json({"a": [True]}, {"r": [1]}) == {"a": [True]}
    assert condense_json({"a": [1]}, {"r": [1]}) == {"a": {"$": "r"}}


def test_top_level_value_can_be_replaced() -> None:
    condensed = condense_json(SCHEMA, {"s": SCHEMA})
    assert condensed == {"$": "s"}
    assert uncondense_json(condensed, {"s": SCHEMA}) == SCHEMA
