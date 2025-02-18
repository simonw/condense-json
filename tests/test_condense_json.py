from condense_json import condense_json


def test_condense_json():
    input_json = {
        "foo": {
            "bar": {
                "string": "This is a string with foxes in it",
                "nested": {
                    "more": ["Here is a string", "another with foxes in it too"]
                },
            }
        }
    }

    replacements = {"1": "with foxes in it"}

    expected_output = {
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


def test_no_replacements():
    input_json = {"text": "This is a normal string"}
    replacements = {"1": "not in the text"}
    expected_output = {"text": "This is a normal string"}

    assert condense_json(input_json, replacements) == expected_output


def test_empty_json():
    input_json = {}
    replacements = {"1": "anything"}
    expected_output = {}

    assert condense_json(input_json, replacements) == expected_output


def test_multiple_replacements():
    input_json = {
        "sentence": "The quick brown fox jumps over the lazy dog",
        "nested": {"list": ["fast fox", "lazy dog", "just some text"]},
    }

    replacements = {"1": "quick brown fox", "2": "lazy dog"}

    expected_output = {
        "sentence": {"$r": ["The ", {"$": "1"}, " jumps over the ", {"$": "2"}]},
        "nested": {"list": ["fast fox", {"$": "2"}, "just some text"]},
    }

    assert condense_json(input_json, replacements) == expected_output


def test_nested_replacements():
    input_json = {
        "outer": {"inner": {"deep": "something deep inside with foxes in it"}}
    }

    replacements = {"1": "with foxes in it"}

    expected_output = {
        "outer": {"inner": {"deep": {"$r": ["something deep inside ", {"$": "1"}]}}}
    }

    assert condense_json(input_json, replacements) == expected_output
