from condense_json import condense_json, uncondense_json
from typing import Dict, List, Mapping, Optional


def test_narrow_concrete_types_accepted_without_annotation() -> None:
    # Covariant input: narrowly typed variables must pass the type
    # checker with no JSONValue annotation required
    narrow_dict: Dict[str, str] = {"s": "a fox"}
    assert condense_json(narrow_dict, {"1": "fox"}) == {
        "s": {"$r": ["a ", {"$": "1"}]}
    }
    narrow_list: List[str] = ["a fox", "no match"]
    assert condense_json(narrow_list, {"1": "fox"}) == [
        {"$r": ["a ", {"$": "1"}]},
        "no match",
    ]
    inferred = {"messages": [{"role": "user", "content": "hi"}]}
    assert condense_json(inferred, {"1": "fox"}) == inferred


def test_results_support_structural_access() -> None:
    # Any out: indexing and len() on results must type-check without
    # isinstance narrowing or casts
    result = uncondense_json({"x": {"$": "1"}, "y": "z"}, {"1": "fox"})
    assert result["x"] == "fox"
    assert len(result) == 2


def test_mixed_scalar_types_accepted() -> None:
    value = {"nested": [1, 2.5, True, None, "text"]}
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
