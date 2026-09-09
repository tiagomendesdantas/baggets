"""S3 artifact store, against a stub client.

No network, no credentials, no moto: the store is a thin translation layer over
boto3 calls, so what is worth testing is that it builds the right keys and does
not mangle arrays on the round trip.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from baggets.aws import S3ArtifactStore


class _StubS3:
    """Enough of the boto3 S3 client to exercise the store."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple] = []

    def put_object(self, Bucket, Key, Body):  # noqa: N803 - boto3 casing
        self.calls.append(("put", Bucket, Key))
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        self.calls.append(("get", Bucket, Key))
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):  # noqa: N803
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def get_paginator(self, _name):
        stub = self

        class _P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [{"Key": k} for k in stub.objects if k.startswith(Prefix)]}

        return _P()


@pytest.fixture
def store():
    return S3ArtifactStore("my-bucket", prefix="baggets/m3", client=_StubS3())


class TestKeys:
    def test_prefix_is_applied_once(self, store):
        assert store._key("monthly/N1402.npz") == "baggets/m3/monthly/N1402.npz"

    def test_a_leading_slash_does_not_double_up(self, store):
        assert store._key("/monthly/x.npz") == "baggets/m3/monthly/x.npz"

    def test_uri_is_built_for_the_caller(self, store):
        assert store.uri("a.npz") == "s3://my-bucket/baggets/m3/a.npz"

    def test_an_empty_prefix_is_not_a_leading_slash(self):
        assert S3ArtifactStore("b", client=_StubS3())._key("a.npz") == "a.npz"


class TestRoundTrip:
    def test_arrays_survive_intact(self, store):
        a = np.arange(24, dtype=np.float64).reshape(4, 6)
        b = np.array([0.5, 1.5])
        store.put_arrays("pool.npz", forecasts=a, errors=b)
        back = store.get_arrays("pool.npz")
        np.testing.assert_array_equal(back["forecasts"], a)
        np.testing.assert_array_equal(back["errors"], b)

    def test_exists_is_false_before_and_true_after(self, store):
        assert not store.exists("x.npz")
        store.put_bytes("x.npz", b"data")
        assert store.exists("x.npz")

    def test_listing_returns_keys_relative_to_the_prefix(self, store):
        store.put_bytes("monthly/a.npz", b"1")
        store.put_bytes("monthly/b.npz", b"2")
        store.put_bytes("yearly/c.npz", b"3")
        assert sorted(store.list_keys("monthly")) == ["monthly/a.npz", "monthly/b.npz"]
