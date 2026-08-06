# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
"""Materialise the litgpt ``TextFiles`` token cache as its own pipeline step.

Added by the Reproducible AI Campaign (row 009). This file does not change any
litgpt behaviour; it only moves work that ``litgpt pretrain`` would otherwise do
inline into a separate, separately-recorded step.

Why
---
``litgpt pretrain --data TextFiles`` tokenizes inside the training process, via
litdata's ``optimize()``. ``optimize()`` always spawns worker processes -- it
derives its own worker count from ``os.cpu_count()``, and the ``--data.num_workers``
flag does *not* control it (that flag only reaches the StreamingDataLoader).

A step that spawns Python worker processes loses the parent process's recorded
pip environment in our provenance capture: the training step is recorded with
none of ``torch``, ``lightning``, ``litdata`` or ``torchmetrics``, which is
precisely the environment a rebuild needs. Measured on litgpt 0.5.13: 10
packages recorded with the inline tokenization, 48 without it.

``TextFiles.prepare_data()`` is a no-op when the output directories already
exist, so running it here leaves the training step free of worker processes and
it records its real environment.

Consistency
-----------
The arguments MUST match the corresponding ``litgpt pretrain`` flags. litgpt
calls::

    data.connect(tokenizer=tokenizer,
                 batch_size=train.micro_batch_size,
                 max_seq_length=model.max_seq_length)

and the cached chunks are laid out for exactly that block size, so a mismatch
would silently produce a cache the training step cannot use correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from litgpt.data import TextFiles
from litgpt.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_path", type=Path, required=True)
    parser.add_argument("--val_data_path", type=Path, required=True)
    parser.add_argument("--tokenizer_dir", type=Path, required=True)
    parser.add_argument(
        "--micro_batch_size", type=int, required=True, help="must equal --train.micro_batch_size"
    )
    parser.add_argument(
        "--max_seq_length", type=int, required=True, help="must equal --train.max_seq_length"
    )
    args = parser.parse_args()

    data = TextFiles(train_data_path=args.train_data_path, val_data_path=args.val_data_path)
    data.connect(
        tokenizer=Tokenizer(args.tokenizer_dir),
        batch_size=args.micro_batch_size,
        max_seq_length=args.max_seq_length,
    )
    data.prepare_data()
    print(f"tokenized -> {data.out_path_train} and {data.out_path_val}")


if __name__ == "__main__":
    main()
