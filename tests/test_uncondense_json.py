from condense_json import condense_json, uncondense_json


def test_uncondense_basic():
    original = {
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
    condensed = condense_json(original, replacements)
    # Now uncondense should recover the original
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_uncondense_non_condensed():
    # If the object is not condensed (no markers), it should remain unchanged.
    original = {"text": "This is a normal string without any changes."}
    replacements = {"1": "not in the text"}
    uncondensed = uncondense_json(original, replacements)
    assert uncondensed == original


def test_uncondense_multiple_replacements():
    original = {
        "sentence": "The quick brown fox jumps over the lazy dog",
        "nested": {"list": ["fast fox", "lazy dog", "just some text"]},
    }
    replacements = {"1": "quick brown fox", "2": "lazy dog"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original


def test_uncondense_mixed():
    # Mixed object where only some strings were condensed.
    original = {
        "a": "Hello world!",
        "b": "Greetings from the quick brown fox",
        "c": {"d": ["No change here", "Another quick brown fox example"]},
    }
    replacements = {"1": "quick brown fox"}
    condensed = condense_json(original, replacements)
    uncondensed = uncondense_json(condensed, replacements)
    assert uncondensed == original
