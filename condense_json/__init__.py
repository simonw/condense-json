import re
from typing import Any, Mapping, Optional, Sequence, Union

# Any value that can be represented in JSON. Uses covariant container
# types, so narrowly typed values such as dict[str, str] are accepted
# without needing a broader annotation
JSONInput = Union[
    str, int, float, bool, None, "Sequence[JSONInput]", "Mapping[str, JSONInput]"
]

# Single-key dicts using one of these keys have special meaning in the
# condensed format, so any such dict found in the input must be escaped.
_MARKER_KEYS = ("$", "$r", "$raw")


class UncondenseError(ValueError):
    """
    Raised by uncondense_json when the condensed object is malformed or
    references a replacement ID not present in the replacements dict.
    """


def condense_json(
    obj: JSONInput, replacements: Mapping[str, Optional[str]]
) -> Any:
    """
    Recursively search through every string in the JSON-like object `obj`.
    For any string that contains one or more of the replacement substrings,
    break the string into segments and replace each found occurrence with
    a dict of the form {"$": replacement_id}. The overall string becomes:
        {"$r": [ "text before", {"$": replacement_id}, "text after", ... ]}

    For example, with:
        obj = {
            "foo": {
                "bar": {
                    "string": "This is a string with foxes in it",
                    "nested": {
                        "more": ["Here is a string", "another with foxes in it too"]
                    }
                }
            }
        }
    and
        replacements = {"1": "with foxes in it"}

    The result will be:
        {
          "foo": {
            "bar": {
              "string": {"$r": ["This is a string ", {"$": "1"}]},
              "nested": {
                "more": [
                  "Here is a string",
                  {"$r": ["another ", {"$": "1"}, " too"]}
                ]
              }
            }
          }
        }

    Matches are found scanning left to right; where replacement substrings
    overlap, the longest match wins regardless of the order of the
    `replacements` dict.

    Any single-key dict in the input whose sole key is "$", "$r" or "$raw"
    would be misinterpreted by uncondense_json, so it is escaped by wrapping
    it in {"$raw": ...}. uncondense_json removes exactly one wrapper layer,
    guaranteeing a lossless round-trip.
    """
    # Filter out any blank replacements
    filtered: dict[str, str] = {
        rep_id: substr for rep_id, substr in replacements.items() if substr
    }

    # If multiple IDs share the same substring, the first one wins
    substr_to_id: dict[str, str] = {}
    for rep_id, substr in filtered.items():
        substr_to_id.setdefault(substr, rep_id)
    # Longer substrings first, so overlapping replacements prefer the
    # longest match regardless of dict insertion order
    pattern: Optional[re.Pattern[str]] = (
        re.compile(
            "|".join(
                re.escape(substr)
                for substr in sorted(filtered.values(), key=len, reverse=True)
            )
        )
        if filtered
        else None
    )

    def process(value: JSONInput) -> Any:
        if isinstance(value, dict):
            processed = {key: process(val) for key, val in value.items()}
            if len(value) == 1 and next(iter(value)) in _MARKER_KEYS:
                return {"$raw": processed}
            return processed
        elif isinstance(value, list):
            return [process(item) for item in value]
        elif isinstance(value, str):
            if pattern is None or not pattern.search(value):
                return value

            segments: list[Any] = []
            last_index: int = 0
            for match in pattern.finditer(value):
                start, end = match.start(), match.end()
                if start > last_index:
                    segments.append(value[last_index:start])
                matched_text: str = match.group(0)
                replacement_id: str = substr_to_id[matched_text]
                segments.append({"$": replacement_id})
                last_index = end
            if last_index < len(value):
                segments.append(value[last_index:])

            # If the entire string was replaced with just {"$": id}, return it directly
            if len(segments) == 1 and isinstance(segments[0], dict):
                return segments[0]

            return {"$r": segments}
        else:
            return value

    return process(obj)


def uncondense_json(
    obj: JSONInput, replacements: Mapping[str, Optional[str]]
) -> Any:
    """
    Recursively reverses the transformation made by condense_json.

    It looks for objects of the form:
      - {"$": replacement_id}  -> replaced entirely, so substitute with replacements[replacement_id]
      - {"$r": [ ... segments ... ]} -> rebuild the string by replacing any {"$": rep_id} segments
        with the actual replacement text.
      - {"$raw": ...} -> an escaped marker-shaped dict from the original input;
        one wrapper layer is removed and the contents are restored without
        being interpreted as a marker.

    Other types (lists, dicts without "$r", or regular strings) are left intact.

    Raises UncondenseError if a marker references an unknown (or blank)
    replacement ID, or if a "$r" structure is malformed.
    """
    # Blank replacements are filtered during condensing, so no valid marker
    # can reference them - treat them as unknown IDs here
    filtered: dict[str, str] = {
        rep_id: substr for rep_id, substr in replacements.items() if substr
    }

    def lookup(rep_id: JSONInput) -> str:
        if not isinstance(rep_id, str) or rep_id not in filtered:
            raise UncondenseError("Unknown replacement id: {!r}".format(rep_id))
        return filtered[rep_id]

    def process(value: JSONInput) -> Any:
        if isinstance(value, dict):
            # Check if this dict represents a condensed string:
            if "$raw" in value and len(value) == 1:
                # Escaped dict: unwrap one layer, and do not treat the
                # top level of the unwrapped value as a marker.
                raw = value["$raw"]
                if isinstance(raw, dict):
                    return {k: process(v) for k, v in raw.items()}
                return process(raw)
            elif "$" in value and len(value) == 1:
                # Short form: the entire string was replaced.
                return lookup(value["$"])
            elif "$r" in value and len(value) == 1:
                # Long form: a list of segments.
                segments = value["$r"]
                if not isinstance(segments, list):
                    raise UncondenseError(
                        '"$r" value must be a list of segments, got: {!r}'.format(
                            segments
                        )
                    )
                rebuilt = ""
                for seg in segments:
                    if isinstance(seg, str):
                        rebuilt += seg
                    elif isinstance(seg, dict) and len(seg) == 1 and "$" in seg:
                        rebuilt += lookup(seg["$"])
                    else:
                        raise UncondenseError(
                            'Invalid "$r" segment: {!r}'.format(seg)
                        )
                return rebuilt
            else:
                # Not a condensed string; process the dict normally.
                return {k: process(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process(item) for item in value]
        else:
            return value

    return process(obj)
