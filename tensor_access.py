"""
Shared accessor layer so scan.py, tamper.py, and app.py's disarm step can
operate on either a real nn.Module (our own demo models) or a plain
state_dict (an uploaded .pt file, which has no .named_parameters() and no
.data to assign into) without branching on which one they were given.
"""
import numpy as np
import torch


def iter_named_tensors(model_or_state_dict):
    """(name, tensor) pairs for either an nn.Module or a plain
    name->tensor mapping (e.g. an uploaded state_dict)."""
    if hasattr(model_or_state_dict, "named_parameters"):
        yield from model_or_state_dict.named_parameters()
    else:
        yield from model_or_state_dict.items()


def get_array(tensor_like):
    if hasattr(tensor_like, "detach"):
        return tensor_like.detach().numpy()
    return np.asarray(tensor_like)


def set_tensor(container, name, new_array):
    new_tensor = torch.from_numpy(new_array.copy())
    if hasattr(container, "named_parameters"):
        dict(container.named_parameters())[name].data = new_tensor
    else:
        container[name] = new_tensor
