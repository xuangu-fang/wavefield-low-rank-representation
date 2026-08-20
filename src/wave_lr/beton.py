"""Minimal reader for FFCV ``.beton`` files.

WaveBench ships its time-harmonic datasets in FFCV's container format. The
``ffcv`` package itself needs a compiled toolchain, but the container is simple
and fully described by its header: fixed-size records point at byte offsets in
the same file. That means a *prefix* of the file is enough to read every sample
whose payload falls inside it, so a 10 GB member can be sampled without being
downloaded whole.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HEADER_TYPE = np.dtype(
    [
        ("version", "<u2"),
        ("num_fields", "<u2"),
        ("page_size", "<u4"),
        ("num_samples", "<u8"),
        ("alloc_table_ptr", "<u8"),
    ],
    align=True,
)

FIELD_DESC_TYPE = np.dtype(
    [("type_id", "<u1"), ("name", ("<u1", 16)), ("arguments", ("<u1", (1024,)))],
    align=True,
)

NDARRAY_ARGS_TYPE = np.dtype([("shape", "<u8", 32), ("type_length", "<u8")])
NDARRAY_TYPE_ID = 4


class BetonReader:
    """Read fixed-size NDArray fields out of a (possibly truncated) beton file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.file_size = self.path.stat().st_size
        header = np.fromfile(self.path, dtype=HEADER_TYPE, count=1)[0]
        self.version = int(header["version"])
        self.num_fields = int(header["num_fields"])
        self.page_size = int(header["page_size"])
        self.num_samples = int(header["num_samples"])
        self.alloc_table_ptr = int(header["alloc_table_ptr"])

        descriptors = np.fromfile(
            self.path, dtype=FIELD_DESC_TYPE, count=self.num_fields,
            offset=HEADER_TYPE.itemsize,
        )
        self.fields = []
        for descriptor in descriptors:
            name = bytes(descriptor["name"]).split(b"\x00")[0].decode("ascii")
            type_id = int(descriptor["type_id"])
            if type_id != NDARRAY_TYPE_ID:
                raise NotImplementedError(
                    f"field {name!r} has type id {type_id}; only NDArray is supported"
                )
            arguments = descriptor["arguments"]
            args = arguments[: NDARRAY_ARGS_TYPE.itemsize].view(NDARRAY_ARGS_TYPE)[0]
            length = int(args["type_length"])
            described = json.loads(
                arguments[NDARRAY_ARGS_TYPE.itemsize :][:length].tobytes().decode("ascii")
            )
            dtype = np.dtype([tuple(entry) for entry in described])["f0"]
            shape = [int(value) for value in args["shape"] if value != 0]
            self.fields.append({"name": name, "dtype": dtype, "shape": tuple(shape)})

        # Every NDArray field stores a single 8-byte pointer per sample.
        metadata_type = np.dtype(
            [("", "<u8") for _ in self.fields], align=True
        )
        self.metadata = np.fromfile(
            self.path, dtype=metadata_type, count=self.num_samples,
            offset=HEADER_TYPE.itemsize + descriptors.nbytes,
        )

    def readable_samples(self) -> np.ndarray:
        """Indices whose payloads lie entirely within the available bytes."""

        ok = np.ones(len(self.metadata), dtype=bool)
        for index, field in enumerate(self.fields):
            nbytes = int(np.prod(field["shape"])) * field["dtype"].itemsize
            pointers = self.metadata[self.metadata.dtype.names[index]]
            ok &= (pointers > 0) & (pointers + nbytes <= self.file_size)
        return np.flatnonzero(ok)

    def read(self, index: int) -> dict[str, np.ndarray]:
        sample = {}
        with open(self.path, "rb") as handle:
            for position, field in enumerate(self.fields):
                pointer = int(self.metadata[self.metadata.dtype.names[position]][index])
                count = int(np.prod(field["shape"]))
                handle.seek(pointer)
                buffer = handle.read(count * field["dtype"].itemsize)
                if len(buffer) < count * field["dtype"].itemsize:
                    raise EOFError(f"sample {index} field {field['name']} beyond file end")
                sample[field["name"]] = np.frombuffer(
                    buffer, dtype=field["dtype"], count=count
                ).reshape(field["shape"])
        return sample

    def describe(self) -> dict:
        return {
            "version": self.version,
            "num_samples": self.num_samples,
            "page_size": self.page_size,
            "fields": [
                {"name": f["name"], "shape": f["shape"], "dtype": str(f["dtype"])}
                for f in self.fields
            ],
            "file_size": self.file_size,
            "complete": self.file_size >= self.alloc_table_ptr,
        }
