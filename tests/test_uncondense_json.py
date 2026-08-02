import pytest
from condense_json import condense_json, uncondense_json, UncondenseError
from typing import Dict, Any


def test_uncondense_basic() -> None:
    original: Dict[str, Any] = {
        "foo": {
            "bar": {
                "string": "This is a string with foxes in it",
                "nested": {
                    "more": ["Here is a string", "another with foxes in it too"]
                },
            }
        }
    }
    replacements: Dict[str, str] = {"1": "with foxes in it"}
    condensed = condense_json(original, replacements)
    # Now uncondense should recover the original
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_uncondense_non_condensed() -> None:
    # If the object is not condensed (no markers), it should remain unchanged.
    original: Dict[str, str] = {"text": "This is a normal string without any changes."}
    replacements: Dict[str, str] = {"1": "not in the text"}
    uncondensed = uncondense_json(original, replacements)
    assert uncondensed == original


def test_uncondense_multiple_replacements() -> None:
    original: Dict[str, Any] = {
        "sentence": "The quick brown fox jumps over the lazy dog",
        "nested": {"list": ["fast fox", "lazy dog", "just some text"]},
    }
    replacements: Dict[str, str] = {"1": "quick brown fox", "2": "lazy dog"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_uncondense_mixed() -> None:
    # Mixed object where only some strings were condensed.
    original: Dict[str, Any] = {
        "a": "Hello world!",
        "b": "Greetings from the quick brown fox",
        "c": {"d": ["No change here", "Another quick brown fox example"]},
    }
    replacements: Dict[str, str] = {"1": "quick brown fox"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_uncondense_error_is_a_value_error() -> None:
    assert issubclass(UncondenseError, ValueError)


def test_unknown_replacement_id_raises() -> None:
    with pytest.raises(UncondenseError) as excinfo:
        uncondense_json({"query": {"$": "gt"}}, {"1": "with foxes"})
    assert "gt" in str(excinfo.value)


def test_unknown_replacement_id_in_segments_raises() -> None:
    with pytest.raises(UncondenseError) as excinfo:
        uncondense_json(
            {"s": {"$r": ["hello ", {"$": "missing"}]}}, {"1": "with foxes"}
        )
    assert "missing" in str(excinfo.value)


def test_non_string_replacement_id_raises() -> None:
    with pytest.raises(UncondenseError):
        uncondense_json({"x": {"$": 100}}, {"1": "with foxes"})


def test_non_list_segments_raises() -> None:
    with pytest.raises(UncondenseError):
        uncondense_json({"x": {"$r": "hello"}}, {"1": "with foxes"})


def test_invalid_segment_type_raises() -> None:
    with pytest.raises(UncondenseError) as excinfo:
        uncondense_json({"x": {"$r": ["a", ["b", "c"]]}}, {"1": "with foxes"})
    assert "segment" in str(excinfo.value)


def test_multi_key_dict_segment_raises() -> None:
    with pytest.raises(UncondenseError):
        uncondense_json(
            {"x": {"$r": [{"$": "1", "extra": "key"}]}}, {"1": "with foxes"}
        )


def test_blank_replacement_id_raises() -> None:
    # condense_json never emits markers for blank replacements, so a marker
    # referencing one is malformed
    with pytest.raises(UncondenseError):
        uncondense_json({"x": {"$": "2"}}, {"1": "with foxes", "2": None})
    with pytest.raises(UncondenseError):
        uncondense_json({"x": {"$": "3"}}, {"1": "with foxes", "3": ""})


def test_escaped_markers_do_not_raise() -> None:
    # A $raw-escaped marker-shaped dict is data, not a marker, so its
    # contents must not be validated as markers
    original = {"query": {"$": "gt"}, "weird": {"$r": "not a list"}}
    replacements: Dict[str, str] = {"1": "with foxes"}
    condensed = condense_json(original, replacements)
    assert uncondense_json(condensed, replacements) == original
