"""Property-based tests: the invariants that must hold for every input.

The central contract is uncondense_json(condense_json(obj, r), r) == obj
for any JSON-safe obj and any replacements mapping. Hypothesis generates
documents with adversarial shapes (marker-like keys, merge-reference
lookalikes) and replacements derived from the documents themselves so
that string, structural and merge matching all actually fire.
"""

import copy
import json
from typing import Any

from hypothesis import given, settings, strategies as st

from condense_json import UncondenseError, condense_json, uncondense_json


def assert_equivalent(a: Any, b: Any) -> None:
    """Equality that Python == cannot fake.

    == alone would let a bool/int swap slip through (True == 1), so
    also compare canonical JSON forms, where they serialize differently.
    sort_keys neutralizes the legitimate key-order changes that
    structural and merge substitution are documented to make.
    """
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# Keys biased toward the marker vocabulary so escaping and merge-shaped
# data get exercised far more often than random text would manage
keys = st.one_of(
    st.sampled_from(["$", "$r", "$raw", "m", "u", "d", "a", "b", "key"]),
    st.text(max_size=8),
)

# Values that Python == conflates but canonical JSON distinguishes -
# drawn often, so equality-semantics bugs cannot hide in rarity
confusables = st.sampled_from([True, False, 0, 1, 0.0, 1.0, "", "0", "1"])

scalars = st.one_of(
    confusables,
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=30),
)

json_values = st.recursive(
    scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(keys, children, max_size=5),
    ),
    max_leaves=25,
)

replacement_values = st.one_of(
    st.text(max_size=20),
    st.dictionaries(keys, scalars, max_size=6),
    st.lists(scalars, max_size=6),
)

replacements_strategy = st.dictionaries(
    st.text(min_size=1, max_size=6), replacement_values, max_size=5
)


def _subtrees(obj: Any, containers: list, strings: list) -> None:
    if isinstance(obj, dict):
        if obj:
            containers.append(obj)
        for value in obj.values():
            _subtrees(value, containers, strings)
    elif isinstance(obj, list):
        if obj:
            containers.append(obj)
        for value in obj:
            _subtrees(value, containers, strings)
    elif isinstance(obj, str) and obj:
        strings.append(obj)


@st.composite
def doc_with_derived_replacements(draw: st.DrawFn) -> tuple:
    """A document plus replacements sampled from its own content.

    Containers are deep-copied and dicts key-reversed, so matching runs
    on structural equality rather than object identity or key order.
    """
    doc = draw(json_values)
    containers: list[Any] = []
    strings: list[str] = []
    _subtrees(doc, containers, strings)
    replacements = {}
    if containers:
        picks = draw(st.lists(st.sampled_from(containers), max_size=3))
        for i, node in enumerate(picks):
            node = copy.deepcopy(node)
            if isinstance(node, dict):
                node = dict(reversed(list(node.items())))
            replacements[f"c{i}"] = node
    if strings:
        picks = draw(st.lists(st.sampled_from(strings), max_size=2))
        for i, text in enumerate(picks):
            replacements[f"s{i}"] = text
    # A few unrelated entries mixed in, which must never affect anything
    replacements.update(draw(replacements_strategy))
    return doc, replacements


@st.composite
def doc_with_merge_bases(draw: st.DrawFn) -> tuple:
    """Documents built from near-copies of a base dict, to force the
    merge path: keys removed, values changed, keys added."""
    base = draw(
        st.dictionaries(keys, scalars, min_size=3, max_size=8).filter(
            lambda d: not (len(d) == 1 and next(iter(d)) in ("$", "$r", "$raw"))
        )
    )
    variants = []
    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        variant = copy.deepcopy(base)
        for key in draw(st.lists(st.sampled_from(sorted(base)), max_size=2)):
            if draw(st.booleans()):
                variant.pop(key, None)
            else:
                variant[key] = draw(scalars)
        for _ in range(draw(st.integers(min_value=0, max_value=2))):
            variant[draw(keys)] = draw(json_values)
        variants.append(variant)
    doc = {"results": variants, "extra": draw(json_values)}
    return doc, {"base": base}


@settings(deadline=None)
@given(json_values, replacements_strategy)
def test_round_trip_arbitrary(doc: Any, replacements: dict) -> None:
    condensed = condense_json(doc, replacements)
    assert_equivalent(uncondense_json(condensed, replacements), doc)


@settings(deadline=None)
@given(doc_with_derived_replacements())
def test_round_trip_with_matching_replacements(pair: tuple) -> None:
    doc, replacements = pair
    condensed = condense_json(doc, replacements)
    assert_equivalent(uncondense_json(condensed, replacements), doc)


@settings(deadline=None)
@given(doc_with_merge_bases())
def test_round_trip_through_merge_encodings(pair: tuple) -> None:
    doc, replacements = pair
    condensed = condense_json(doc, replacements)
    assert_equivalent(uncondense_json(condensed, replacements), doc)


@settings(deadline=None)
@given(doc_with_derived_replacements())
def test_condensed_output_is_json_serializable(pair: tuple) -> None:
    doc, replacements = pair
    json.dumps(condense_json(doc, replacements))


@settings(deadline=None)
@given(doc_with_derived_replacements())
def test_repeated_application_still_round_trips(pair: tuple) -> None:
    # The README promises losslessness "including when applied more
    # than once" - double condense must unwind under double uncondense
    doc, replacements = pair
    twice = condense_json(condense_json(doc, replacements), replacements)
    once = uncondense_json(twice, replacements)
    assert_equivalent(uncondense_json(once, replacements), doc)


@settings(deadline=None)
@given(json_values, replacements_strategy)
def test_uncondense_never_raises_anything_but_uncondense_error(
    doc: Any, replacements: dict
) -> None:
    # Arbitrary input treated as a condensed document: either it
    # resolves or it fails with the typed error, never anything else
    try:
        uncondense_json(doc, replacements)
    except UncondenseError:
        pass
