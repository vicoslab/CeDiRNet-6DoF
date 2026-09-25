"""Loading policy for trusted training and evaluation checkpoints."""
import inspect

import torch


def load_trusted_checkpoint(path, *, map_location):
    """Load a checkpoint produced by this project or another trusted source.

    Training snapshots include NumPy logger scalars and optimizer/history state,
    not just tensor weights. They require pickle loading (weights_only=False),
    which can execute code: never use this helper with an untrusted checkpoint.
    Always remap tensors to the caller's device, including for CPU-only runs.
    """
    kwargs = {'map_location': map_location}
    # Older PyTorch forwards unknown options to pickle instead of accepting
    # weights_only. Inspect the API rather than retrying deserialization errors.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)
