import torch
import spconv.pytorch as spconv

def dense_collate(batch, device="cuda"):
    batch = [item for item in batch if item is not None]

    imgs, mjjs, labels = zip(*batch)

    batch_imgs = torch.stack(imgs).to(device)     # (B, C, H, W)
    batch_mjj  = torch.stack(mjjs).to(device)     # (B,)
    batch_lab  = torch.stack(labels).to(device)   # (B,)

    return batch_imgs, batch_mjj, batch_lab


def dense_to_sparse(batch_imgs, device="cuda"):
    """
    batch_imgs: (B, C, H, W)
    returns: SparseConvTensor
    """

    B, C, H, W = batch_imgs.shape

    # active if ANY channel non-zero
    mask = torch.any(batch_imgs != 0, dim=1)  # (B, H, W)

    nz = torch.nonzero(mask, as_tuple=False)
    # nz: (N_active, 3) -> (batch, eta, phi)

    features = batch_imgs[
        nz[:, 0], :, nz[:, 1], nz[:, 2]
    ]  # (N_active, C)

    return spconv.SparseConvTensor(
        features=features.to(device),
        indices=nz.to(device, dtype=torch.int32),
        spatial_shape=(H, W),
        batch_size=B,
    )



