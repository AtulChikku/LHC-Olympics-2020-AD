import torch
import spconv.pytorch as spconv
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
sys.path.append("/home/asm/LHC-AD/Attempt 3/Preprocessing")

from Dataset_class import PrecomputedJetDataset
from dense_to_sparse import dense_collate, dense_to_sparse
from Feature_extractor import SparseFeatureExtractor
from kT_declustering import build_sequence, get_global_representation

device = "cuda"
jet_path = "/home/asm/LHC-AD/Working files/3C_jet_images_parallel.h5"

output_dir = "./precomputed_sequences_3"
os.makedirs(output_dir, exist_ok=True)

pt_threshold_frac = 0.008
batch_size = 32
chunk_size = 5000  # sequences per file

def dense_collate_wrapper(batch):
        return dense_collate(batch, device=device)


def build_sequences(indices, tag):
    dataset = PrecomputedJetDataset(jet_path, indices=indices)
    MAX_SEQ_LEN = 30

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dense_collate_wrapper,
        multiprocessing_context="spawn",
        num_workers=6,        
    )

    feature_extractor = SparseFeatureExtractor(
        input_channels=2, out_channels=64
    ).to(device)

    feature_extractor.eval()
    for p in feature_extractor.parameters():
        p.requires_grad = False

    sequences = []
    file_idx = 0

    for batch_imgs, batch_mjj, labels in tqdm(loader, desc=f"Building {tag}"):
        
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_imgs = batch_imgs[:, :2]  # drop log(pT)
        sparse = dense_to_sparse(batch_imgs)
        indices_sp, feats = feature_extractor(sparse)

        B = batch_imgs.shape[0]

        for b in range(B):
            mask = indices_sp[:, 0] == b
            if mask.sum() < 3:
                continue

            coords_int = indices_sp[mask][:, 1:].long()
            feats_b = feats[mask]

            jet_pt = batch_imgs[
                b, 0,
                coords_int[:, 0],
                coords_int[:, 1]
            ]

            keep = jet_pt > pt_threshold_frac * jet_pt.max()
            if keep.sum() < 3:
                continue

            # apply keep mask ONCE
            coords = coords_int[keep].float()
            feats_b = feats_b[keep]
            jet_pt = jet_pt[keep]

            # cap max sequence length
            if jet_pt.numel() > MAX_SEQ_LEN:
                topk = torch.topk(jet_pt, MAX_SEQ_LEN).indices
                coords = coords[topk]
                feats_b = feats_b[topk]
                jet_pt = jet_pt[topk]

            # normalize features
            feats_b = feats_b / (feats_b.norm(dim=1, keepdim=True) + 1e-6)


            seq = build_sequence(feats_b, coords, jet_pt)
            global_repr = get_global_representation(seq)
            

            sequences.append({
                "sequence": global_repr.cpu(),
                "label": labels[b].item(),
                "mjj": batch_mjj[b].item()
            })

            if tag == "sig_test" and file_idx == 0 and len(sequences) < 10:
                print("Signal labels seen:", set(s["label"] for s in sequences))

            if len(sequences) >= chunk_size:
                out_path = os.path.join(
                    output_dir, f"{tag}_{file_idx:03d}.pt"
                )
                torch.save(sequences, out_path)
                sequences.clear()
                file_idx += 1

    # save remainder
    if sequences:
        out_path = os.path.join(
            output_dir, f"{tag}_{file_idx:03d}.pt"
        )
        torch.save(sequences, out_path)

def main():
    bg_train_indices = list(range(0, 200000))
    bg_test_indices  = list(range(300000, 400000))
    sig_test_indices = list(range(1000000, 1050000))

    build_sequences(bg_train_indices, "bg_train")
    build_sequences(bg_test_indices, "bg_test")
    build_sequences(sig_test_indices, "sig_test")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
