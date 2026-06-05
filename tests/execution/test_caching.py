import asyncio

from comfy_execution import caching


class _StubDynPrompt:
    def __init__(self, nodes):
        self._nodes = nodes

    def has_node(self, node_id):
        return node_id in self._nodes

    def get_node(self, node_id):
        return self._nodes[node_id]


class _StubIsChangedCache:
    async def get(self, node_id):
        return None


class _StubNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


def test_shallow_is_changed_signature_keeps_primitive_only_list_shallow():
    assert caching._shallow_is_changed_signature([1, "two", None, True]) == (
        "is_changed_list",
        (1, "two", None, True),
    )


def test_shallow_is_changed_signature_keeps_primitive_only_tuple_shallow():
    assert caching._shallow_is_changed_signature((1, "two", None, True)) == (
        "is_changed_tuple",
        (1, "two", None, True),
    )


def test_shallow_is_changed_signature_keeps_structured_builtin_fingerprint_list():
    assert caching._shallow_is_changed_signature([("seed", 42), {"cfg": 8}]) == (
        "is_changed_list",
        (
            ("tuple", ("seed", 42)),
            ("dict", (("cfg", 8),)),
        ),
    )


def test_shallow_is_changed_signature_does_not_use_to_hashable(monkeypatch):
    monkeypatch.setattr(
        caching,
        "to_hashable",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("is_changed signature must not deep-canonicalize")
        ),
    )

    signature = caching._shallow_is_changed_signature([("seed", 42), {"cfg": 8}])

    assert signature == (
        "is_changed_list",
        (
            ("tuple", ("seed", 42)),
            ("dict", (("cfg", 8),)),
        ),
    )


def test_get_immediate_node_signature_canonicalizes_non_link_inputs(monkeypatch):
    live_value = [1, {"nested": [2, 3]}]
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_immediate_node_signature(dynprompt, "1", {}))

    assert signature == (
        "TestCacheNode",
        None,
        ("value", ("list", (1, ("dict", (("nested", ("list", (2, 3))),))))),
    )


def test_to_hashable_walks_dicts_without_rebinding_traversal_stack():
    live_value = {
        "outer": {"nested": [2, 3]},
        "items": [{"leaf": 4}],
    }

    assert caching.to_hashable(live_value) == (
        "dict",
        (
            ("items", ("list", (("dict", (("leaf", 4),)),))),
            ("outer", ("dict", (("nested", ("list", (2, 3))),))),
        ),
    )


def test_get_immediate_node_signature_fails_closed_for_opaque_non_link_input(monkeypatch):
    class OpaqueRuntimeValue:
        pass

    live_value = OpaqueRuntimeValue()
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_immediate_node_signature(dynprompt, "1", {}))

    assert isinstance(signature, caching.Unhashable)


def test_get_node_signature_propagates_unhashable_immediate_fragment(monkeypatch):
    class OpaqueRuntimeValue:
        pass

    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": OpaqueRuntimeValue()},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_node_signature(dynprompt, "1"))

    assert isinstance(signature, caching.Unhashable)


def test_get_node_signature_never_visits_raw_non_link_input(monkeypatch):
    live_value = [1, 2, 3]
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})
    monkeypatch.setattr(
        caching,
        "_signature_to_hashable",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("outer signature canonicalizer should not run")
        ),
    )

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_node_signature(dynprompt, "1"))

    assert isinstance(signature, tuple)


def test_get_node_signature_keeps_deep_canonicalized_input_fragment(monkeypatch):
    live_value = 1
    for _ in range(8):
        live_value = [live_value]
    expected = caching.to_hashable(live_value)

    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_node_signature(dynprompt, "1"))

    assert isinstance(signature, tuple)
    assert signature[0][2][0] == "value"
    assert signature[0][2][1] == expected


def test_get_node_signature_keeps_large_precanonicalized_fragment(monkeypatch):
    live_value = object()
    canonical_fragment = ("tuple", tuple(("list", (index, index + 1)) for index in range(256)))
    dynprompt = _StubDynPrompt(
        {
            "1": {
                "class_type": "TestCacheNode",
                "inputs": {"value": live_value},
            }
        }
    )

    monkeypatch.setitem(caching.nodes.NODE_CLASS_MAPPINGS, "TestCacheNode", _StubNode)
    monkeypatch.setattr(caching, "NODE_CLASS_CONTAINS_UNIQUE_ID", {})
    monkeypatch.setattr(
        caching,
        "to_hashable",
        lambda value, max_nodes=caching._MAX_SIGNATURE_CONTAINER_VISITS: (
            canonical_fragment if value is live_value else caching.Unhashable()
        ),
    )
    monkeypatch.setattr(
        caching,
        "_signature_to_hashable",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("outer signature canonicalizer should not run")
        ),
    )

    keyset = caching.CacheKeySetInputSignature(dynprompt, [], _StubIsChangedCache())
    signature = asyncio.run(keyset.get_node_signature(dynprompt, "1"))

    assert isinstance(signature, tuple)
    assert signature[0][2] == ("value", canonical_fragment)
