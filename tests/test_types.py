from condense_json import JSONValue, condense_json, uncondense_json
from typing import Mapping, Optional


def test_jsonvalue_is_exported() -> None:
    value: JSONValue = {"nested": [1, 2.5, True, None, "text"]}
    assert condense_json(value, {}) == value


def test_top_level_list() -> None:
    result: JSONValue = condense_json(["a fox ran", "no match"], {"1": "fox"})
    assert result == [{"$r": ["a ", {"$": "1"}, " ran"]}, "no match"]
    assert uncondense_json(result, {"1": "fox"}) == ["a fox ran", "no match"]


def test_top_level_string() -> None:
    assert condense_json("just a fox", {"1": "fox"}) == {
        "$r": ["just a ", {"$": "1"}]
    }
    assert condense_json("fox", {"1": "fox"}) == {"$": "1"}
    assert uncondense_json({"$": "1"}, {"1": "fox"}) == "fox"


def test_replacements_accepts_read_only_mapping_with_optional_values() -> None:
    replacements: Mapping[str, Optional[str]] = {"1": "fox", "2": None, "3": ""}
    condensed = condense_json({"s": "a fox"}, replacements)
    assert condensed == {"s": {"$r": ["a ", {"$": "1"}]}}
    assert uncondense_json(condensed, replacements) == {"s": "a fox"}
