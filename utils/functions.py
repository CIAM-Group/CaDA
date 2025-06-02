import math
import torch
import pickle
import numpy as np
from torch import Tensor
from tensordict import TensorDict
from einops import rearrange
from typing import Optional, Union

from utils.utils import *


def _batchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    # example: x: bs, return: bs*repeat
    s = x.shape # = td.batch_size
    out = x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:]) # (s[0] * repeats, *s[1:]) = (repeats*bs)
    # out = x.expand(repeats, *s).reshape(s[0] * repeats, *s[1:])
    return out


def batchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x

def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x

def unbatchify_and_gather(x: Tensor, idx: Tensor, n: int):
    """first unbatchify a tensor by n and then gather (usually along the unbatchified dimension)
    by the specified index
    """
    x = unbatchify(x, n)
    return gather_by_index(x, idx, dim=idx.dim())


def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)


def save_tensordict_to_npz(tensordict, filename, compress: bool = False):
    """Save a TensorDict to a npz file
    We assume that the TensorDict contains a dictionary of tensors
    """
    x_dict = {k: v.numpy() for k, v in tensordict.items()}
    if compress:
        np.savez_compressed(filename, **x_dict)
    else:
        np.savez(filename, **x_dict)

def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)

def get_distance(x: Tensor, y: Tensor):
    """Euclidean distance between two tensors of shape `[..., n, dim]`"""
    return (x - y).norm(p=2, dim=-1)

def clip_grad_norms(param_groups, max_norm=math.inf):
    # optimizer.param_groups
    # Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def get_opt_sol_path(dir, problem, size):
    all_opt_sol = {
        'CVRP': {50: 'hgs_cvrp50_uniform.pkl', 100: 'hgs_cvrp100_uniform.pkl'},
        'OVRP': {50: 'or_tools_200s_ovrp50_uniform.pkl', 100: 'lkh_ovrp100_uniform.pkl'},
        'VRPB': {50: 'or_tools_200s_vrpb50_uniform.pkl', 100: 'or_tools_400s_vrpb100_uniform.pkl'},
        'VRPL': {50: 'or_tools_200s_vrpl50_uniform.pkl', 100: 'lkh_vrpl100_uniform.pkl'},
        'VRPTW': {50: 'hgs_vrptw50_uniform.pkl', 100: 'hgs_vrptw100_uniform.pkl'},
        'OVRPTW': {50: 'or_tools_200s_ovrptw50_uniform.pkl', 100: 'or_tools_400s_ovrptw100_uniform.pkl'},
        'OVRPB': {50: 'or_tools_200s_ovrpb50_uniform.pkl', 100: 'or_tools_400s_ovrpb100_uniform.pkl'},
        'OVRPL': {50: 'or_tools_200s_ovrpl50_uniform.pkl', 100: 'or_tools_400s_ovrpl100_uniform.pkl'},
        'VRPBL': {50: 'or_tools_200s_vrpbl50_uniform.pkl', 100: 'or_tools_400s_vrpbl100_uniform.pkl'},
        'VRPBTW': {50: 'or_tools_200s_vrpbtw50_uniform.pkl', 100: 'or_tools_400s_vrpbtw100_uniform.pkl'},
        'VRPLTW': {50: 'or_tools_200s_vrpltw50_uniform.pkl', 100: 'or_tools_400s_vrpltw100_uniform.pkl'},
        'OVRPBL': {50: 'or_tools_200s_ovrpbl50_uniform.pkl', 100: 'or_tools_400s_ovrpbl100_uniform.pkl'},
        'OVRPBTW': {50: 'or_tools_200s_ovrpbtw50_uniform.pkl', 100: 'or_tools_400s_ovrpbtw100_uniform.pkl'},
        'OVRPLTW': {50: 'or_tools_200s_ovrpltw50_uniform.pkl', 100: 'or_tools_400s_ovrpltw100_uniform.pkl'},
        'VRPBLTW': {50: 'or_tools_200s_vrpbltw50_uniform.pkl', 100: 'or_tools_400s_vrpbltw100_uniform.pkl'},
        'OVRPBLTW': {50: 'or_tools_200s_ovrpbltw50_uniform.pkl', 100: 'or_tools_400s_ovrpbltw100_uniform.pkl'},
    }
    return os.path.join(dir, all_opt_sol[problem][size])

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def load_dataset(filename, disable_print=False):
    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
    if not disable_print:
        print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
    return data