from condense_json import condense_json, uncondense_json
from typing import Dict, Any


def test_raw_escape_roundtrip() -> None:
    """Test that $raw content remains intact through a complete roundtrip."""
    original: Dict[str, Any] = {
        "text": "This is a string with foxes in it",
        "special_markers": {"$raw": {"$": "1"}},
    }
    replacements = {"1": "with foxes in it"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_dollar_key_is_escaped() -> None:
    original: Dict[str, Any] = {"price": {"$": "100"}}
    replacements: Dict[str, str] = {"1": "with foxes"}
    condensed = condense_json(original, replacements)
    assert condensed == {"price": {"$raw": {"$": "100"}}}
    assert uncondense_json(condensed, replacements) == original


def test_dollar_r_key_is_escaped() -> None:
    original: Dict[str, Any] = {"note": {"$r": ["hello", "world"]}}
    replacements: Dict[str, str] = {"1": "with foxes"}
    condensed = condense_json(original, replacements)
    assert condensed == {"note": {"$raw": {"$r": ["hello", "world"]}}}
    assert uncondense_json(condensed, replacements) == original


def test_strings_inside_escaped_marker_are_still_condensed() -> None:
    original: Dict[str, Any] = {"$": "a string with foxes in it"}
    replacements: Dict[str, str] = {"1": "with foxes in it"}
    condensed = condense_json(original, replacements)
    assert condensed == {"$raw": {"$": {"$r": ["a string ", {"$": "1"}]}}}
    assert uncondense_json(condensed, replacements) == original


def test_nested_marker_shaped_dicts() -> None:
    original: Dict[str, Any] = {"$": {"$": "a"}}
    replacements: Dict[str, str] = {"1": "with foxes"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_double_condense_roundtrip() -> None:
    original: Dict[str, Any] = {
        "text": "This is a string with foxes in it",
        "price": {"$": "100"},
    }
    replacements: Dict[str, str] = {"1": "with foxes in it"}
    condensed_twice = condense_json(condense_json(original, replacements), replacements)
    uncondensed_twice = uncondense_json(
        uncondense_json(condensed_twice, replacements), replacements
    )
    assert uncondensed_twice == original


def test_escaping_applies_even_with_no_replacements() -> None:
    original: Dict[str, Any] = {"query": {"$": "gt"}}
    condensed = condense_json(original, {})
    assert condensed == {"query": {"$raw": {"$": "gt"}}}
    assert uncondense_json(condensed, {}) == original


def test_multi_key_dicts_are_not_escaped() -> None:
    original: Dict[str, Any] = {"data": {"$": "gt", "other": "value"}}
    replacements: Dict[str, str] = {"1": "with foxes"}
    condensed = condense_json(original, replacements)
    assert condensed == original
    assert uncondense_json(condensed, replacements) == original


def test_escaped_markers_inside_lists() -> None:
    original: Dict[str, Any] = {
        "items": [{"$": "a"}, {"$r": "b"}, {"$raw": "c"}, "plain string"]
    }
    replacements: Dict[str, str] = {"1": "with foxes"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original
