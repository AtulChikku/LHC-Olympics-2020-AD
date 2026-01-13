import math
import torch

def delta_phi(a, b):
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi


def kt_ordering(coords, pt):
    """
    coords: (N, 2) -> (eta, phi)
    pt:     (N,)
    returns: list of indices
    """

    remaining = list(range(len(pt)))
    ordered = []

    current = torch.argmax(pt).item()
    ordered.append(current)
    remaining.remove(current)

    while remaining:
        i = current

        def kt_dist(j):
            dR = torch.sqrt(
                (coords[i, 0] - coords[j, 0]) ** 2 +
                delta_phi(coords[i, 1], coords[j, 1]) ** 2
            )
            return min(pt[i] ** 2, pt[j] ** 2) * dR ** 2

        current = max(remaining, key=kt_dist)
        ordered.append(current)
        remaining.remove(current)

    return ordered


def build_sequence(jet_feats, jet_coords, jet_pt):
    """
    jet_feats:  (N, D)
    jet_coords: (N, 2)
    jet_pt:     (N,)
    """

    DELTA_ETA = 0.1 / 50   # or your binning
    DELTA_PHI = 2*(math.pi) / 50

    # convert pixel indices → centered coords
    coords = jet_coords.clone()
    coords[:, 0] = (coords[:, 0] - 25) * DELTA_ETA
    coords[:, 1] = (coords[:, 1] - 25) * DELTA_PHI

    order = kt_ordering(coords, jet_pt)

    seq = []
    for i in order:
        seq.append(
            torch.cat([
                jet_pt[i].unsqueeze(0),
                jet_coords[i],
                jet_feats[i],
            ])
        )

    return torch.stack(seq)

def compute_symmetry_scores(coords, pt):
    # 1. pT asymmetry entropy
    a = pt / (pt.sum() + 1e-6)
    pt_entropy = -(a * torch.log(a + 1e-6)).sum()

    # 2. mean and max split scale
    dR = []
    for i in range(len(pt)-1):
        j = coords[i]
        k = coords[i+1]
        dR.append(torch.norm(j-k).item())
    dR = torch.tensor(dR)
    return {
        "pt_entropy": pt_entropy,
        "mean_dR": dR.mean().item(),
        "max_dR":  dR.max().item(),
        "dR_var":  dR.var().item(),
    }


def get_kt_order(jet_coords, jet_pt):
    return kt_ordering(jet_coords, jet_pt)  # your existing function


def get_global_representation(sequence):
    """
    Input: (T, 67) sequence from your kT_declustering logic
    Output: (134,) fixed-size vector
    """
    # Mean across the particles
    mean_feat = torch.mean(sequence, dim=0)
    # Max across the particles 
    max_feat = torch.max(sequence, dim=0)[0]
    
    # Concatenate to get a 134-dim jet descriptor
    return torch.cat([mean_feat, max_feat])
