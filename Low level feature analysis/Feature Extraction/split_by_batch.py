import torch

def split_by_batch(indices, features, batch_size):
    """
    indices: (N, 3)
    features: (N, D)
    returns: list of length B, each (N_i, D)
    """

    out = [[] for _ in range(batch_size)]

    for idx, feat in zip(indices, features):
        b = idx[0].item()
        out[b].append(feat)

    return [torch.stack(v) if len(v) else None for v in out]
