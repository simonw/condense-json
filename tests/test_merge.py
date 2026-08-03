"""Merge references: encoding a dict as a base object plus a patch."""

import json
from typing import Any

import pytest

from condense_json import UncondenseError, condense_json, uncondense_json

ENV: dict[str, Any] = {
    "object": "response",
    "service_tier": "default",
    "status": "completed",
    "truncation": "disabled",
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "store": False,
    "top_logprobs": 0,
    "tools": [],
}


def test_mostly_matching_dict_becomes_base_plus_patch() -> None:
    obj = dict(ENV, id="resp_123", usage={"input_tokens": 9})
    condensed = condense_json(obj, {"env": ENV})
    assert condensed == {
        "$": {"m": "env", "u": {"id": "resp_123", "usage": {"input_tokens": 9}}}
    }
    assert uncondense_json(condensed, {"env": ENV}) == obj


def test_changed_key_lands_in_the_patch() -> None:
    obj = dict(ENV, status="in_progress")
    condensed = condense_json(obj, {"env": ENV})
    assert condensed == {"$": {"m": "env", "u": {"status": "in_progress"}}}
    assert uncondense_json(condensed, {"env": ENV}) == obj


def test_missing_base_keys_become_deletions() -> None:
    obj = {k: v for k, v in ENV.items() if k != "tools"}
    condensed = condense_json(obj, {"env": ENV})
    assert condensed == {"$": {"m": "env", "d": ["tools"]}}
    assert uncondense_json(condensed, {"env": ENV}) == obj


def test_exact_match_still_uses_the_short_form() -> None:
    condensed = condense_json(dict(ENV), {"env": ENV})
    assert condensed == {"$": "env"}


def test_patch_values_are_condensed_recursively() -> None:
    details = {"cached_tokens": 0, "cache_write_tokens": 0, "padding": "p" * 40}
    obj = dict(ENV, usage={"input_tokens": 4, "input_tokens_details": details})
    replacements = {"env": ENV, "details": details}
    condensed = condense_json(obj, replacements)
    assert condensed["$"]["u"]["usage"]["input_tokens_details"] == {"$": "details"}
    assert uncondense_json(condensed, replacements) == obj


def test_unrelated_dict_prices_itself_out() -> None:
    obj = {"name": "Cleo", "species": "dog"}
    assert condense_json(obj, {"env": ENV}) == obj


def test_plain_form_wins_when_patch_would_be_bigger() -> None:
    # Only two short keys shared; the deletion list alone outweighs
    # writing the dict out plainly
    obj: dict[str, Any] = {"status": "completed", "store": False}
    assert condense_json(obj, {"env": ENV}) == obj


def test_best_of_multiple_bases_is_chosen() -> None:
    near = dict(ENV, extra="value")
    obj = dict(ENV, extra="value", id="x")
    condensed = condense_json(obj, {"far": {"status": "completed"}, "near": near})
    assert condensed == {"$": {"m": "near", "u": {"id": "x"}}}
    assert (
        uncondense_json(condensed, {"far": {"status": "completed"}, "near": near})
        == obj
    )


def test_key_order_never_matters() -> None:
    reordered = dict(reversed(list(ENV.items())))
    condensed = condense_json(dict(reordered, id="x"), {"env": ENV})
    assert condensed == {"$": {"m": "env", "u": {"id": "x"}}}


def test_scalar_type_confusion_is_not_equality() -> None:
    # store False in the base, 0 in the object: different in canonical
    # form, so the key must travel in the patch
    obj = dict(ENV, store=0)
    condensed = condense_json(obj, {"env": ENV})
    assert condensed["$"]["u"] == {"store": 0}
    result = uncondense_json(condensed, {"env": ENV})
    assert result["store"] == 0 and result["store"] is not False


def test_input_containing_a_merge_shaped_marker_round_trips() -> None:
    obj: dict[str, Any] = {"payload": {"$": {"m": "env", "u": {"id": "fake"}}}}
    replacements: dict[str, Any] = {"other": {"a": 1, "b": 2, "c": 3}}
    condensed = condense_json(obj, replacements)
    assert condensed == {"payload": {"$raw": {"$": {"m": "env", "u": {"id": "fake"}}}}}
    assert uncondense_json(condensed, replacements) == obj


def test_merged_results_are_independent_copies() -> None:
    condensed: dict[str, Any] = {
        "a": {"$": {"m": "env", "u": {"id": "x"}}},
        "b": {"$": {"m": "env"}},
    }
    replacements = {"env": ENV}
    result = uncondense_json(condensed, replacements)
    result["a"]["tools"].append("mutated")
    assert result["b"]["tools"] == []
    assert ENV["tools"] == []


def test_uncondense_rejects_malformed_merge_references() -> None:
    replacements: dict[str, Any] = {"env": ENV, "s": "some string value here"}
    bad_refs: list[dict[str, Any]] = [
        {"$": {"u": {"a": 1}}},  # no base
        {"$": {"m": "missing"}},  # unknown base
        {"$": {"m": "s"}},  # base is a string replacement
        {"$": {"m": "env", "x": 1}},  # unknown field
        {"$": {"m": "env", "d": "tools"}},  # d not a list
        {"$": {"m": "env", "d": ["nope"]}},  # deleting a key the base lacks
        {"$": {"m": "env", "u": [1]}},  # u not a dict
    ]
    for bad in bad_refs:
        with pytest.raises(UncondenseError):
            uncondense_json(bad, replacements)


def test_merge_reference_inside_r_segments_is_rejected() -> None:
    condensed: dict[str, Any] = {"text": {"$r": ["x", {"$": {"m": "env"}}]}}
    with pytest.raises(UncondenseError):
        uncondense_json(condensed, {"env": ENV})


def test_round_trip_on_a_payload_shaped_document() -> None:
    payload = dict(
        ENV,
        id="resp_abc",
        model="gpt-5-mini",
        usage={"input_tokens": 256, "output_tokens": 19},
        output=[
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
        ],
    )
    replacements = {"env": ENV}
    condensed = condense_json(payload, replacements)
    assert json.dumps(condensed) != json.dumps(payload)  # something happened
    assert uncondense_json(condensed, replacements) == payload


def test_first_id_wins_for_duplicate_merge_bases() -> None:
    # Two equivalent bases: the one that appears first in the mapping
    # is the one a partial match references, matching the documented
    # first-ID-wins rule - not the one that sorts first.
    # Equal-length IDs, so the byte-cost comparison cannot secretly
    # break the tie in the test's favour
    obj = dict(ENV, id="x")
    replacements: dict[str, Any] = {"zz": dict(ENV), "aa": dict(ENV)}
    condensed = condense_json(obj, replacements)
    assert condensed == {"$": {"m": "zz", "u": {"id": "x"}}}
    assert uncondense_json(condensed, replacements) == obj
