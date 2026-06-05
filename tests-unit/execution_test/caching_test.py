"""Unit tests for cache-signature canonicalization hardening."""

import asyncio
import importlib
import sys
import types

import pytest


class _DummyNode:
    """Minimal node stub used to satisfy cache-signature class lookups."""

    @staticmethod
    def INPUT_TYPES():
        """Return a minimal empty input schema for unit tests."""
        return {"required": {}}


class _FakeDynPrompt:
    """Small DynamicPrompt stand-in with only the methods these tests need."""

    def __init__(self, nodes_by_id):
        """Store test nodes by id."""
        self._nodes_by_id = nodes_by_id

    def has_node(self, node_id):
        """Return whether the fake prompt contains the requested node."""
        return node_id in self._nodes_by_id

    def get_node(self, node_id):
        """Return the stored node payload for the requested id."""
        return self._nodes_by_id[node_id]


class _FakeIsChangedCache:
    """Async stub for `is_changed` lookups used by cache-key generation."""

    def __init__(self, values):
        """Store canned `is_changed` responses keyed by node id."""
        self._values = values

    async def get(self, node_id):
        """Return the canned `is_changed` value for a node."""
        return self._values[node_id]


class _OpaqueValue:
    """Hashable opaque object used to exercise fail-closed unordered hashing paths."""


@pytest.fixture
def caching_module(monkeypatch):
    """Import `comfy_execution.caching` with lightweight stub dependencies."""
    torch_module = types.ModuleType("torch")
    psutil_module = types.ModuleType("psutil")
    nodes_module = types.ModuleType("nodes")
    nodes_module.NODE_CLASS_MAPPINGS = {}
    graph_module = types.ModuleType("comfy_execution.graph")

    class DynamicPrompt:
        """Placeholder graph type so the caching module can import cleanly."""

        pass

    graph_module.DynamicPrompt = DynamicPrompt

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "psutil", psutil_module)
    monkeypatch.setitem(sys.modules, "nodes", nodes_module)
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph_module)
    monkeypatch.delitem(sys.modules, "comfy_execution.caching", raising=False)

    module = importlib.import_module("comfy_execution.caching")
    module = importlib.reload(module)
    return module, nodes_module


def test_signature_to_hashable_handles_shared_builtin_substructures(caching_module):
    """Shared built-in substructures should canonicalize without collapsing to Unhashable."""
    caching, _ = caching_module
    shared = [{"value": 1}, {"value": 2}]

    signature = caching._signature_to_hashable([shared, shared])

    assert signature[0] == "list"
    assert signature[1][0] == signature[1][1]
    assert signature[1][0][0] == "list"
    assert signature[1][0][1][0] == ("dict", (("value", 1),))
    assert signature[1][0][1][1] == ("dict", (("value", 2),))


def test_signature_to_hashable_fails_closed_on_opaque_values(caching_module):
    """Opaque values should collapse the full signature to Unhashable immediately."""
    caching, _ = caching_module

    signature = caching._signature_to_hashable(["safe", object()])

    assert isinstance(signature, caching.Unhashable)


def test_signature_to_hashable_stops_descending_after_failure(caching_module, monkeypatch):
    """Once canonicalization fails, later recursive descent should stop immediately."""
    caching, _ = caching_module
    original = caching._signature_to_hashable_impl
    marker = object()
    marker_seen = False

    def tracking_canonicalize(obj, *args, **kwargs):
        """Track whether recursion reaches the nested marker after failure."""
        nonlocal marker_seen
        if obj is marker:
            marker_seen = True
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_signature_to_hashable_impl", tracking_canonicalize)

    signature = caching._signature_to_hashable([object(), [marker]])

    assert isinstance(signature, caching.Unhashable)
    assert marker_seen is False


def test_signature_to_hashable_snapshots_list_before_recursing(caching_module, monkeypatch):
    """List canonicalization should read a point-in-time snapshot before recursive descent."""
    caching, _ = caching_module
    original = caching._signature_to_hashable_impl
    marker = ("marker",)
    values = [marker, 2]

    def mutating_canonicalize(obj, *args, **kwargs):
        """Mutate the live list during recursion to verify snapshot-based traversal."""
        if obj is marker:
            values[1] = 3
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_signature_to_hashable_impl", mutating_canonicalize)

    signature = caching._signature_to_hashable(values)

    assert signature == ("list", (("tuple", ("marker",)), 2))
    assert values[1] == 3


def test_signature_to_hashable_snapshots_dict_before_recursing(caching_module, monkeypatch):
    """Dict canonicalization should read a point-in-time snapshot before recursive descent."""
    caching, _ = caching_module
    original = caching._signature_to_hashable_impl
    marker = ("marker",)
    values = {"first": marker, "second": 2}

    def mutating_canonicalize(obj, *args, **kwargs):
        """Mutate the live dict during recursion to verify snapshot-based traversal."""
        if obj is marker:
            values["second"] = 3
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_signature_to_hashable_impl", mutating_canonicalize)

    signature = caching._signature_to_hashable(values)

    assert signature == ("dict", (("first", ("tuple", ("marker",))), ("second", 2)))
    assert values["second"] == 3


@pytest.mark.parametrize(
    "container_factory",
    [
        lambda marker: [marker],
        lambda marker: (marker,),
        lambda marker: {marker},
        lambda marker: frozenset({marker}),
        lambda marker: {"key": marker},
    ],
)
def test_signature_to_hashable_fails_closed_on_runtimeerror(caching_module, monkeypatch, container_factory):
    """Traversal RuntimeError should degrade canonicalization to Unhashable."""
    caching, _ = caching_module
    original = caching._signature_to_hashable_impl
    marker = object()

    def raising_canonicalize(obj, *args, **kwargs):
        """Raise a traversal RuntimeError for the marker value and delegate otherwise."""
        if obj is marker:
            raise RuntimeError("container changed during iteration")
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_signature_to_hashable_impl", raising_canonicalize)

    signature = caching._signature_to_hashable(container_factory(marker))

    assert isinstance(signature, caching.Unhashable)


def test_to_hashable_handles_shared_builtin_substructures(caching_module):
    """The legacy helper should still hash sanitized built-ins stably when used directly."""
    caching, _ = caching_module
    shared = [{"value": 1}, {"value": 2}]

    sanitized = [shared, shared]
    hashable = caching.to_hashable(sanitized)

    assert hashable[0] == "list"
    assert hashable[1][0] == hashable[1][1]
    assert hashable[1][0][0] == "list"


def test_to_hashable_uses_parent_snapshot_during_expanded_phase(caching_module, monkeypatch):
    """Expanded-phase assembly should not reread a live parent container after snapshotting."""
    caching, _ = caching_module
    original_sort_key = caching._sanitized_sort_key
    outer = [{"marker"}, 2]

    def mutating_sort_key(obj, *args, **kwargs):
        """Mutate the live parent while a child container is being canonicalized."""
        if obj == "marker":
            outer[1] = 3
        return original_sort_key(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_sanitized_sort_key", mutating_sort_key)

    hashable = caching.to_hashable(outer)

    assert hashable == ("list", (("set", ("marker",)), 2))
    assert outer[1] == 3


def test_to_hashable_fails_closed_for_ordered_container_with_opaque_child(caching_module):
    """Ordered containers should fail closed when a child cannot be canonicalized."""
    caching, _ = caching_module

    result = caching.to_hashable([object()])

    assert isinstance(result, caching.Unhashable)


def test_to_hashable_canonicalizes_dict_insertion_order(caching_module):
    """Dicts with the same content should hash identically regardless of insertion order."""
    caching, _ = caching_module

    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}

    assert caching.to_hashable(first) == ("dict", (("a", 1), ("b", 2)))
    assert caching.to_hashable(first) == caching.to_hashable(second)


def test_to_hashable_fails_closed_for_opaque_dict_key(caching_module):
    """Opaque dict keys should fail closed instead of being traversed during hashing."""
    caching, _ = caching_module

    hashable = caching.to_hashable({_OpaqueValue(): 1})

    assert isinstance(hashable, caching.Unhashable)


@pytest.mark.parametrize(
    "container_factory",
    [
        set,
        frozenset,
    ],
)
def test_to_hashable_fails_closed_on_runtimeerror(caching_module, monkeypatch, container_factory):
    """Traversal RuntimeError should degrade unordered hash conversion to Unhashable."""
    caching, _ = caching_module

    def raising_sort_key(obj, *args, **kwargs):
        """Raise a traversal RuntimeError while unordered values are canonicalized."""
        raise RuntimeError("container changed during iteration")

    monkeypatch.setattr(caching, "_sanitized_sort_key", raising_sort_key)

    hashable = caching.to_hashable(container_factory({"value"}))

    assert isinstance(hashable, caching.Unhashable)


def test_to_hashable_fails_closed_for_ambiguous_dict_ordering(caching_module, monkeypatch):
    """Ambiguous dict key ordering should fail closed instead of using insertion order."""
    caching, _ = caching_module
    original_sort_key = caching._sanitized_sort_key
    ambiguous = {"a": 1, "b": 1}

    def colliding_sort_key(obj, *args, **kwargs):
        """Force two distinct primitive keys to share the same ordering key."""
        if obj == "a" or obj == "b":
            return ("COLLIDE",)
        return original_sort_key(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_sanitized_sort_key", colliding_sort_key)

    hashable = caching.to_hashable(ambiguous)

    assert isinstance(hashable, caching.Unhashable)


def test_signature_to_hashable_fails_closed_for_ambiguous_dict_ordering(caching_module, monkeypatch):
    """Ambiguous dict sort ties should fail closed instead of depending on input order."""
    caching, _ = caching_module
    original_sort_key = caching._primitive_signature_sort_key
    ambiguous = {"a": 1, "b": 1}

    def colliding_sort_key(obj):
        """Force two distinct primitive keys to share the same ordering key."""
        if obj == "a" or obj == "b":
            return ("COLLIDE",)
        return original_sort_key(obj)

    monkeypatch.setattr(caching, "_primitive_signature_sort_key", colliding_sort_key)

    sanitized = caching._signature_to_hashable(ambiguous)

    assert isinstance(sanitized, caching.Unhashable)


def test_signature_to_hashable_fails_closed_for_opaque_dict_key(caching_module):
    """Opaque dict keys should fail closed instead of being recursively canonicalized."""
    caching, _ = caching_module

    sanitized = caching._signature_to_hashable({_OpaqueValue(): 1})

    assert isinstance(sanitized, caching.Unhashable)


def test_signature_to_hashable_fails_closed_on_dict_key_sort_collisions_even_with_distinct_values(caching_module, monkeypatch):
    """Different values must not mask dict key-sort collisions during canonicalization."""
    caching, _ = caching_module
    original_sort_key = caching._primitive_signature_sort_key

    def colliding_sort_key(obj):
        """Force two distinct primitive keys to share the same ordering key."""
        if obj == "a" or obj == "b":
            return ("COLLIDE",)
        return original_sort_key(obj)

    monkeypatch.setattr(caching, "_primitive_signature_sort_key", colliding_sort_key)

    sanitized = caching._signature_to_hashable({"a": 1, "b": 2})

    assert isinstance(sanitized, caching.Unhashable)


@pytest.mark.parametrize(
    "container_factory",
    [
        set,
        frozenset,
    ],
)
def test_to_hashable_fails_closed_for_ambiguous_unordered_values(caching_module, monkeypatch, container_factory):
    """Ambiguous unordered values should fail closed instead of depending on iteration order."""
    caching, _ = caching_module
    original_sort_key = caching._sanitized_sort_key
    container = container_factory({"a", "b"})

    def colliding_sort_key(obj, *args, **kwargs):
        """Force two distinct primitive values to share the same ordering key."""
        if obj == "a" or obj == "b":
            return ("COLLIDE",)
        return original_sort_key(obj, *args, **kwargs)

    monkeypatch.setattr(caching, "_sanitized_sort_key", colliding_sort_key)

    hashable = caching.to_hashable(container)

    assert isinstance(hashable, caching.Unhashable)


def test_get_node_signature_returns_top_level_unhashable_for_tainted_signature(caching_module, monkeypatch):
    """Tainted full signatures should fail closed before `to_hashable()` runs."""
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)
    monkeypatch.setattr(
        caching,
        "to_hashable",
        lambda *_args, **_kwargs: pytest.fail("to_hashable should not run for tainted signatures"),
    )

    is_changed_value = []
    is_changed_value.append(is_changed_value)

    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"value": 5},
            }
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": is_changed_value}),
    )

    signature = asyncio.run(key_set.get_node_signature(dynprompt, "node"))

    assert isinstance(signature, caching.Unhashable)


def test_shallow_is_changed_signature_accepts_primitive_lists(caching_module):
    """Primitive-only `is_changed` lists should stay hashable without deep descent."""
    caching, _ = caching_module

    sanitized = caching._shallow_is_changed_signature([1, "two", None, True])

    assert sanitized == ("is_changed_list", (1, "two", None, True))


def test_shallow_is_changed_signature_accepts_structured_builtin_fingerprint_lists(caching_module):
    """Structured built-in `is_changed` fingerprints should remain representable."""
    caching, _ = caching_module

    sanitized = caching._shallow_is_changed_signature([("seed", 42), {"cfg": 8}])

    assert sanitized == (
        "is_changed_list",
        (
            ("tuple", ("seed", 42)),
            ("dict", (("cfg", 8),)),
        ),
    )


def test_shallow_is_changed_signature_fails_closed_for_opaque_payload(caching_module):
    """Opaque `is_changed` payloads should still fail closed."""
    caching, _ = caching_module

    sanitized = caching._shallow_is_changed_signature([_OpaqueValue()])

    assert isinstance(sanitized, caching.Unhashable)


def test_get_immediate_node_signature_fails_closed_for_unhashable_is_changed(caching_module, monkeypatch):
    """Recursive `is_changed` payloads should fail the full fragment closed."""
    caching, nodes_module = caching_module
    monkeypatch.setitem(nodes_module.NODE_CLASS_MAPPINGS, "UnitTestNode", _DummyNode)

    is_changed_value = []
    is_changed_value.append(is_changed_value)
    dynprompt = _FakeDynPrompt(
        {
            "node": {
                "class_type": "UnitTestNode",
                "inputs": {"value": 5},
            }
        }
    )
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        ["node"],
        _FakeIsChangedCache({"node": is_changed_value}),
    )

    signature = asyncio.run(key_set.get_immediate_node_signature(dynprompt, "node", {}))

    assert isinstance(signature, caching.Unhashable)


def test_get_immediate_node_signature_fails_closed_for_missing_node(caching_module):
    """Missing nodes should return the fail-closed sentinel instead of a NaN tuple."""
    caching, _ = caching_module
    dynprompt = _FakeDynPrompt({})
    key_set = caching.CacheKeySetInputSignature(
        dynprompt,
        [],
        _FakeIsChangedCache({}),
    )

    signature = asyncio.run(key_set.get_immediate_node_signature(dynprompt, "missing", {}))

    assert isinstance(signature, caching.Unhashable)
