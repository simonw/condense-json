import re
from typing import Any, Dict


def condense_json(obj: Dict, replacements: Dict[str, str]) -> Any:
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
    """
    # Filter out any blank replacements
    replacements = {rep_id: substr for rep_id, substr in replacements.items() if substr}

    if not replacements:
        return obj

    substr_to_id = {substr: rep_id for rep_id, substr in replacements.items()}
    pattern = re.compile("|".join(map(re.escape, replacements.values())))

    def process(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: process(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [process(item) for item in value]
        elif isinstance(value, str):
            if not pattern.search(value):
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


def uncondense_json(obj: Dict, replacements: Dict[str, str]) -> Any:
    """
    Recursively reverses the transformation made by condense_json.

    It looks for objects of the form:
      - {"$": replacement_id}  -> replaced entirely, so substitute with replacements[replacement_id]
      - {"$r": [ ... segments ... ]} -> rebuild the string by replacing any {"$": rep_id} segments
        with the actual replacement text.

    Other types (lists, dicts without "$r", or regular strings) are left intact.
    """

    def process(value: Any) -> Any:
        if isinstance(value, dict):
            # Check if this dict represents a condensed string:
            if "$" in value and len(value) == 1:
                # Short form: the entire string was replaced.
                rep_id = value["$"]
                return replacements[rep_id]
            elif "$r" in value and len(value) == 1:
                # Long form: a list of segments.
                segments = value["$r"]
                rebuilt = ""
                for seg in segments:
                    if isinstance(seg, str):
                        rebuilt += seg
                    elif isinstance(seg, dict) and "$" in seg:
                        rep_id = seg["$"]
                        rebuilt += replacements[rep_id]
                    else:
                        # If an unexpected type is encountered, process it recursively.
                        rebuilt += str(process(seg))
                return rebuilt
            else:
                # Not a condensed string; process the dict normally.
                return {k: process(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process(item) for item in value]
        else:
            return value

    return process(obj)
