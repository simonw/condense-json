import copy
import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence, Union

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


def _canonical(value: JSONInput) -> Optional[str]:
    """
    The canonical JSON form used for structural equality: keys sorted,
    compact separators, non-ASCII left as-is. Returns None for values
    that cannot be serialized as JSON, which simply never match.
    """
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except (TypeError, ValueError):
        return None


def _split_replacements(
    replacements: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], set[tuple[type, int]]]:
    """
    Split a replacements mapping into its two participating kinds.

    Returns (string_replacements, canonical_to_id, container_shapes):
    non-empty strings, which match as substrings; and non-empty dicts and
    lists, keyed here by canonical form for structural lookup, with the
    (type, length) shapes present so callers can skip canonicalizing
    nodes that cannot possibly match. Everything else - None, empty
    strings, empty containers, numbers, booleans - is filtered out, the
    same silent treatment blank strings have always had.
    """
    strings: dict[str, str] = {}
    canonical_to_id: dict[str, str] = {}
    shapes: set[tuple[type, int]] = set()
    for rep_id, value in replacements.items():
        if isinstance(value, str):
            if value:
                strings[rep_id] = value
        elif isinstance(value, (dict, list)):
            if value:
                canonical = _canonical(value)
                if canonical is not None:
                    canonical_to_id.setdefault(canonical, rep_id)
                    shapes.add((type(value), len(value)))
    return strings, canonical_to_id, shapes


def condense_json(obj: JSONInput, replacements: Mapping[str, Any]) -> Any:
    """
    Recursively search through the JSON-like object `obj`, replacing
    content that matches an entry in `replacements` with a reference to
    its ID.

    String replacement values match as substrings. For any string in
    `obj` that contains one or more of them, the string is broken into
    segments and each found occurrence becomes a dict of the form
    {"$": replacement_id}. The overall string becomes:
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

    Dict and list replacement values match *structurally*: any subtree of
    `obj` that is equal to the replacement value - compared in canonical
    JSON form, so key order never matters - is replaced whole with
    {"$": replacement_id}. Matching is outermost-wins: once a subtree
    matches, its interior is not searched further. With:
        replacements = {"schema": {"type": "object", "properties": {...}}}
    any occurrence of that schema as a subtree condenses to
    {"$": "schema"}. Structural matching is strictly structural - a
    string that happens to contain the JSON serialization of the value
    is not matched, because a reference inside a string must resolve to
    a string.

    Dict replacements also serve as *merge bases*: a dict node that is
    mostly equal to a base is encoded as the base plus a patch,
        {"$": {"m": base_id, "u": {added or changed keys}, "d": [absent keys]}}
    whenever that form is smaller than writing the node out - a cost
    comparison decides, so unrelated bases price themselves out. Patch
    values in "u" are condensed recursively. Deletion is the explicit
    "d" key list, never a null sentinel, because null is a legitimate
    payload value.

    If multiple IDs share the same value, the first one wins. Values that
    are None, empty, or non-string scalars are ignored.

    Any single-key dict in the input whose sole key is "$", "$r" or "$raw"
    would be misinterpreted by uncondense_json, so it is escaped by wrapping
    it in {"$raw": ...}. uncondense_json removes exactly one wrapper layer,
    guaranteeing a lossless round-trip. A subtree that matches a structural
    replacement is referenced rather than escaped, whatever its shape.
    """
    filtered, canonical_to_id, container_shapes = _split_replacements(replacements)

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

    # Dict replacements double as merge bases: a dict node that is
    # *mostly* equal to a base can be encoded as the base plus a patch.
    # Kept in mapping order with equivalent values deduplicated, so the
    # first ID wins - the same rule every other replacement kind follows.
    merge_bases: dict[str, dict] = {}
    _seen_base_canonicals: set[str] = set()
    for rep_id, val in replacements.items():
        if isinstance(val, dict) and val:
            base_canonical = _canonical(val)
            if base_canonical is None or base_canonical in _seen_base_canonicals:
                continue
            _seen_base_canonicals.add(base_canonical)
            merge_bases[rep_id] = val
    # Canonical forms of every base value, computed once per call - the
    # bases are fixed, so per-node recomputation would be pure waste.
    base_canonicals: dict[str, dict[str, Optional[str]]] = {
        rep_id: {key: _canonical(val) for key, val in base.items()}
        for rep_id, base in merge_bases.items()
    }
    # Canonical forms of input subtrees, memoized by object identity for
    # the duration of this call (the document keeps every node alive, so
    # ids are stable). A node's canonical form is otherwise recomputed
    # once per base that considers it.
    canonical_memo: dict[int, Optional[str]] = {}

    def canonical_of(value: JSONInput) -> Optional[str]:
        key = id(value)
        if key not in canonical_memo:
            canonical_memo[key] = _canonical(value)
        return canonical_memo[key]

    def structural_id(value: JSONInput) -> Optional[str]:
        # The shape check keeps this cheap: most nodes are not even the
        # same type and length as any replacement, so their canonical
        # form is never computed.
        if (type(value), len(value)) not in container_shapes:  # type: ignore[arg-type]
            return None
        canonical = canonical_of(value)
        if canonical is None:
            return None
        return canonical_to_id.get(canonical)

    def _size(value: Any) -> Optional[int]:
        # None for values that cannot be serialized - condense has
        # always tolerated JSON-ish input, so sizing must not introduce
        # a crash; an unsizable form simply never wins.
        try:
            return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
        except (TypeError, ValueError):
            return None

    def best_merge(raw: dict, processed: dict, plain: Any) -> Optional[dict]:
        """The cheapest {"$": {"m": id, "u": ..., "d": ...}} encoding of
        ``raw``, or None when writing it out plainly is smaller.

        ``u`` holds keys added or changed relative to the base (their
        already-condensed forms), ``d`` lists base keys the node lacks.
        A base that shares little with the node produces a patch bigger
        than the node itself, so it prices itself out - no applicability
        heuristic needed beyond a small overlap prefilter.
        """
        # Cheap overlap check first: most nodes share keys with no base
        # at all, and skipping them avoids ever serializing their plain
        # form for sizing - that serialization at every level is what
        # would otherwise make deeply nested documents quadratic.
        viable = [
            (rep_id, base)
            for rep_id, base in merge_bases.items()
            if len(raw.keys() & base.keys()) >= 2
        ]
        if not viable:
            return None
        plain_cost = _size(plain)
        if plain_cost is None:
            return None
        best: Optional[dict] = None
        best_cost = plain_cost
        for rep_id, base in viable:
            canonicals = base_canonicals[rep_id]
            u = {
                key: processed[key]
                for key in raw
                if key not in base
                or canonicals[key] is None
                or canonicals[key] != canonical_of(raw[key])
            }
            d = [key for key in base if key not in raw]
            ref: dict[str, Any] = {"m": rep_id}
            if u:
                ref["u"] = u
            if d:
                ref["d"] = d
            candidate = {"$": ref}
            cost = _size(candidate)
            if cost is not None and cost < best_cost:
                best, best_cost = candidate, cost
        return best

    def process(value: JSONInput) -> Any:
        if isinstance(value, (dict, list)) and container_shapes:
            # Outermost-wins: a matched subtree is referenced whole, and
            # its interior is not searched. Checked before marker
            # escaping, on the raw value the caller supplied.
            rep_id = structural_id(value)
            if rep_id is not None:
                return {"$": rep_id}
        if isinstance(value, dict):
            processed = {key: process(val) for key, val in value.items()}
            plain: Any = processed
            if len(value) == 1 and next(iter(value)) in _MARKER_KEYS:
                plain = {"$raw": processed}
            if merge_bases:
                merged = best_merge(value, processed, plain)
                if merged is not None:
                    return merged
            return plain
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


def uncondense_json(obj: JSONInput, replacements: Mapping[str, Any]) -> Any:
    """
    Recursively reverses the transformation made by condense_json.

    It looks for objects of the form:
      - {"$": replacement_id}  -> replaced entirely, so substitute with replacements[replacement_id].
        String replacements substitute as strings; dict and list replacements
        substitute as an independent deep copy of the value, so mutating the
        result never aliases the replacements mapping or other markers.
      - {"$": {"m": base_id, "u": {...}, "d": [...]}} -> a merge against a dict
        replacement: the base is deep-copied, "d" keys removed, "u" entries
        applied on top (processed recursively, so they may contain markers).
      - {"$r": [ ... segments ... ]} -> rebuild the string by replacing any {"$": rep_id} segments
        with the actual replacement text. Only string replacements may appear
        here: a reference inside a string must resolve to a string.
      - {"$raw": ...} -> an escaped marker-shaped dict from the original input;
        one wrapper layer is removed and the contents are restored without
        being interpreted as a marker.

    Other types (lists, dicts without "$r", or regular strings) are left intact.

    Raises UncondenseError if a marker references an unknown (or blank)
    replacement ID, if a "$r" segment references a non-string replacement,
    or if a "$r" structure is malformed.
    """
    # Values that condense_json filters out can never be referenced by a
    # valid marker - treat them as unknown IDs here
    strings, _, _ = _split_replacements(replacements)
    containers: dict[str, Any] = {
        rep_id: value
        for rep_id, value in replacements.items()
        if isinstance(value, (dict, list)) and value
    }

    def lookup(rep_id: JSONInput) -> Any:
        if isinstance(rep_id, str):
            if rep_id in strings:
                return strings[rep_id]
            if rep_id in containers:
                return copy.deepcopy(containers[rep_id])
        raise UncondenseError("Unknown replacement id: {!r}".format(rep_id))

    def lookup_string(rep_id: JSONInput) -> str:
        if isinstance(rep_id, dict):
            raise UncondenseError(
                'Merge reference is not valid in a "$r" segment: {!r}'.format(rep_id)
            )
        value = lookup(rep_id)
        if not isinstance(value, str):
            raise UncondenseError(
                'Non-string replacement id in "$r" segment: {!r}'.format(rep_id)
            )
        return value

    def apply_merge(ref: dict, process: "Callable[[JSONInput], Any]") -> dict:
        """Rebuild a dict from {"m": base_id, "u": updates, "d": deletions}.

        The base is deep-copied, ``d`` keys removed, ``u`` entries
        processed (they may contain markers of their own) and applied on
        top. Deletion is an explicit key list, never a null sentinel -
        payloads contain legitimate nulls.
        """
        extra = set(ref) - {"m", "u", "d"}
        if extra or "m" not in ref:
            raise UncondenseError("Invalid merge reference: {!r}".format(ref))
        base_id = ref["m"]
        base = containers.get(base_id) if isinstance(base_id, str) else None
        if not isinstance(base, dict):
            raise UncondenseError(
                "Merge base must be a dict replacement: {!r}".format(base_id)
            )
        result = copy.deepcopy(base)
        deletions = ref.get("d", [])
        if not isinstance(deletions, list):
            raise UncondenseError('"d" must be a list of keys: {!r}'.format(deletions))
        for key in deletions:
            if not isinstance(key, str) or key not in result:
                raise UncondenseError(
                    "Merge deletion of a key the base does not have: {!r}".format(key)
                )
            del result[key]
        updates = ref.get("u", {})
        if not isinstance(updates, dict):
            raise UncondenseError('"u" must be a dict of updates: {!r}'.format(updates))
        for key, val in updates.items():
            result[key] = process(val)
        return result

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
                # Short form: the entire value was replaced - or, when
                # the reference is a dict, a merge against a base object.
                if isinstance(value["$"], dict):
                    return apply_merge(value["$"], process)
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
                        rebuilt += lookup_string(seg["$"])
                    else:
                        raise UncondenseError('Invalid "$r" segment: {!r}'.format(seg))
                return rebuilt
            else:
                # Not a condensed string; process the dict normally.
                return {k: process(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process(item) for item in value]
        else:
            return value

    return process(obj)
