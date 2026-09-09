"""S3 artifact store for the Stage-A forecast cache and model checkpoints.

Stage A of the pipeline is the expensive half: bootstrapping a pool and fitting
every member, per series, is hours of compute that Stage B then reuses for free
across strategies. That cache is worth moving off one laptop — this module puts
it on S3 so a run can resume on another machine, or fan out across several.

boto3 is an optional extra (``pip install baggets[aws]``). Credentials come from
the standard boto3 chain (env, shared config, instance role); this module never
takes a key as an argument and never writes one anywhere.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["S3ArtifactStore"]

_BOTO3_HINT = (
    "The S3 artifact store needs boto3, an optional extra of baggets.\n"
    "Install it with:  uv sync --extra aws   (or: pip install 'baggets[aws]')"
)


def _boto3():
    """Import boto3 lazily so the package works without it."""
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via message
        raise ModuleNotFoundError(_BOTO3_HINT) from exc
    return boto3


class S3ArtifactStore:
    """Read/write baggets artifacts under one ``s3://bucket/prefix``.

    Keys are relative to the prefix, so callers pass ``"m3/monthly/N1402.npz"``
    and never build URIs by hand.
    """

    def __init__(self, bucket: str, prefix: str = "", client: Any | None = None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = _boto3().client("s3")
        return self._client

    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"

    # ---------------------------------------------------------------- basics
    def exists(self, key: str) -> bool:
        """True if the object is there. Used to skip already-computed series."""
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "403"):
                return False
            raise
        return True

    def list_keys(self, key_prefix: str = "") -> Iterator[str]:
        """Yield keys under ``key_prefix``, relative to the store prefix."""
        full = self._key(key_prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        head = len(self.prefix) + 1 if self.prefix else 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                yield obj["Key"][head:]

    def put_bytes(self, key: str, payload: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=payload)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()

    # ------------------------------------------------------------- artifacts
    def put_file(self, key: str, path: str | Path) -> str:
        """Upload a Stage-A ``.npz`` cache file as produced by ``precompute_series``."""
        self.client.upload_file(str(path), self.bucket, self._key(key))
        return self.uri(key)

    def get_file(self, key: str, path: str | Path) -> Path:
        """Download an artifact to ``path``, creating parent directories."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(path))
        return path

    def put_arrays(self, key: str, **arrays: np.ndarray) -> str:
        """Store named arrays as a compressed ``.npz`` object, without touching disk."""
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return self.put_bytes(key, buf.getvalue())

    def get_arrays(self, key: str) -> dict[str, np.ndarray]:
        with np.load(io.BytesIO(self.get_bytes(key))) as z:
            return {name: z[name] for name in z.files}

    # ----------------------------------------------------------- checkpoints
    def put_checkpoint(self, key: str, module: Any) -> str:
        """Store a trained ``torch.nn.Module``'s state dict.

        Weights only — ``torch.save`` of a whole module pickles code paths, and
        an artifact store is exactly the wrong place to keep something whose
        load is arbitrary code execution.
        """
        import torch

        buf = io.BytesIO()
        torch.save(module.state_dict(), buf)
        return self.put_bytes(key, buf.getvalue())

    def get_checkpoint(self, key: str, module: Any) -> Any:
        """Load a state dict from S3 into ``module`` and return it."""
        import torch

        state = torch.load(io.BytesIO(self.get_bytes(key)), weights_only=True)
        module.load_state_dict(state)
        return module
