from condense_json import condense_json
from typing import Dict, Any, List


def test_condense_json() -> None:
    input_json: Dict[str, Any] = {
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

    expected_output: Dict[str, Any] = {
        "foo": {
            "bar": {
                "string": {"$r": ["This is a string ", {"$": "1"}]},
                "nested": {
                    "more": [
                        "Here is a string",
                        {"$r": ["another ", {"$": "1"}, " too"]},
                    ]
                },
            }
        }
    }

    assert condense_json(input_json, replacements) == expected_output


def test_no_replacements() -> None:
    input_json: Dict[str, str] = {"text": "This is a normal string"}
    replacements: Dict[str, str] = {"1": "not in the text"}
    expected_output: Dict[str, str] = {"text": "This is a normal string"}

    assert condense_json(input_json, replacements) == expected_output


def test_replacement_not_used() -> None:
    input = {"messages": [{"role": "user", "content": "What is 1231 * 2331?"}]}
    replacements = {"r:01jv577ycee7re8wqdebbvygys": ""}
    output = condense_json(input, replacements)
    assert output == input


def test_empty_json() -> None:
    input_json: Dict[str, Any] = {}
    replacements: Dict[str, str] = {"1": "anything"}
    expected_output: Dict[str, Any] = {}

    assert condense_json(input_json, replacements) == expected_output


def test_multiple_replacements() -> None:
    input_json: Dict[str, Any] = {
        "sentence": "The quick brown fox jumps over the lazy dog",
        "nested": {"list": ["fast fox", "lazy dog", "just some text"]},
    }

    replacements: Dict[str, str] = {"1": "quick brown fox", "2": "lazy dog"}

    expected_output: Dict[str, Any] = {
        "sentence": {"$r": ["The ", {"$": "1"}, " jumps over the ", {"$": "2"}]},
        "nested": {"list": ["fast fox", {"$": "2"}, "just some text"]},
    }

    assert condense_json(input_json, replacements) == expected_output


def test_nested_replacements() -> None:
    input_json: Dict[str, Any] = {
        "outer": {"inner": {"deep": "something deep inside with foxes in it"}}
    }

    replacements: Dict[str, str] = {"1": "with foxes in it"}

    expected_output: Dict[str, Any] = {
        "outer": {"inner": {"deep": {"$r": ["something deep inside ", {"$": "1"}]}}}
    }

    assert condense_json(input_json, replacements) == expected_output


def test_blank_or_none_replacements() -> None:
    input_json: Dict[str, Any] = {
        "outer": {"inner": {"deep": "something deep inside with foxes in it"}}
    }

    replacements: Dict[str, Any] = {"1": "deep", "2": None, "3": ""}

    expected_output: Dict[str, Any] = {
        "outer": {
            "inner": {
                "deep": {"$r": ["something ", {"$": "1"}, " inside with foxes in it"]}
            }
        }
    }

    assert condense_json(input_json, replacements) == expected_output
