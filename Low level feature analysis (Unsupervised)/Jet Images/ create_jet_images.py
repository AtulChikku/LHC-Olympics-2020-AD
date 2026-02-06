import h5py
import hdf5plugin
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pyjet import cluster, DTYPE_PTEPM
import math


N_FEATURES = 3
N_ETA = 50
N_PHI = 50
MAX_R = 1.1
R_CLUSTERING = 1.0
PTMIN = 500.0

eta_bins = np.linspace(-MAX_R, MAX_R, N_ETA + 1)
phi_bins = np.linspace(-MAX_R, MAX_R, N_PHI + 1)

NUM_CPU_WORKERS = mp.cpu_count() - 1 
MAX_JOBS_QUEUE = 500
H5_CHUNK_SIZE = 100

# --- MJJ Calculation ---
def calculate_MJJ(j1, j2):
    E = j1.e + j2.e
    px = j1.px + j2.px
    py = j1.py + j2.py
    pz = j1.pz + j2.pz
    M2 = E*E - (px*px + py*py + pz*pz)
    return np.sqrt(max(M2, 0.0))


def make_dense_jet_image(row):
    """Generates one dense image and MJJ from a raw row of features."""
    n = len(row) // 3
    pT = row[0:n*3:3]
    eta = row[1:n*3:3]
    phi = row[2:n*3:3]

    vec = np.zeros(n, dtype=DTYPE_PTEPM)
    vec["pT"] = pT
    vec["eta"] = eta
    vec["phi"] = phi
    vec["mass"] = 0.0

    seq = cluster(vec, R=R_CLUSTERING, p=-1)
    jets = seq.inclusive_jets(ptmin=PTMIN)

    if len(jets) < 2:
        return None, None

    J1, J2 = jets[0], jets[1]
    MJJ = calculate_MJJ(J1, J2)

    dense = np.zeros((N_FEATURES, N_ETA, N_PHI), dtype=np.float32)

    jet_pt = J1.pt
    jet_eta = J1.eta
    jet_phi = J1.phi
    jet_E = J1.e

    for c in J1.constituents_array():
        c_pt = c["pT"]
        c_eta = c["eta"]

        d_eta = c_eta - jet_eta
        d_phi = c["phi"] - jet_phi

        if d_phi > np.pi: d_phi -= 2*np.pi
        if d_phi < -np.pi: d_phi += 2*np.pi

        if abs(d_eta) > MAX_R or abs(d_phi) > MAX_R:
            continue

        i = np.digitize(d_eta, eta_bins) - 1
        j = np.digitize(d_phi, phi_bins) - 1
        if i < 0 or i >= N_ETA or j < 0 or j >= N_PHI:
            continue

        # Features
        pT_norm = c_pt / (jet_pt + 1e-9)
        E_norm  = (c_pt * np.cosh(c_eta)) / (jet_E + 1e-9)
        log_norm = np.log(pT_norm + 1e-6)

        dense[0, i, j] += pT_norm
        dense[1, i, j] += E_norm
        dense[2, i, j] += log_norm

    return dense, MJJ

def _worker_process(index, feats, label):
    """Worker function executed in parallel."""
    dense, MJJ = make_dense_jet_image(feats)
    return index, dense, MJJ, label


def preprocess(h5_in_path, out_path):
    print("Opening input file:", h5_in_path)
    
    # Use context managers for safe I/O
    with h5py.File(h5_in_path, "r") as fin:
        rows = fin["df"]["block0_values"]
        N = rows.shape[0]

        with h5py.File(out_path, "a") as f:
            
            # --- Dataset Setup ---
            if "images" not in f:
                print("Creating datasets...")
                f.create_dataset("images", (N, 3, 50, 50), dtype="float32",
                                 chunks=(H5_CHUNK_SIZE, 3, 50, 50), compression=None)
                f.create_dataset("MJJ", (N,), dtype="float32", chunks=(H5_CHUNK_SIZE,))
                f.create_dataset("labels", (N,), dtype="int8", chunks=(H5_CHUNK_SIZE,))
                f.attrs["processed_until"] = 0
            
            start_idx = f.attrs["processed_until"]
            d_images = f["images"]
            d_mjj = f["MJJ"]
            d_labels = f["labels"]

            print(f"Starting parallel processing with {NUM_CPU_WORKERS} workers from index {start_idx}/{N}")

            # --- Parallel Execution ---
            with ProcessPoolExecutor(max_workers=NUM_CPU_WORKERS) as executor:
                futures = []
           
                for i in tqdm(range(start_idx, N), initial=start_idx, total=N):
                    
                    row_data = rows[i]
                    label = int(row_data[-1])
                    feats = row_data[:-1]
                    
                    future = executor.submit(_worker_process, index=i, feats=feats, label=label)
                    futures.append(future)
                    
                    if len(futures) >= MAX_JOBS_QUEUE or i == N - 1:
                        for future in as_completed(futures):
                            idx, dense, MJJ, label = future.result()
                            
                            if dense is not None:
                                d_images[idx] = dense
                                d_mjj[idx] = MJJ
                                d_labels[idx] = label 
                            else:
                                d_mjj[idx] = -1.0
                                d_labels[idx] = -1 # Use -1 to mark skipped labels
                        
                        futures = [] # Clear the list

                    # Update resume pointer every 1000 samples
                    if i % 1000 == 0:
                        f.attrs["processed_until"] = i
                        f.flush()

                f.attrs["processed_until"] = N
                f.flush()

    print("Done. Output saved to:", out_path)


if __name__ == "__main__":
    
    INPUT_FILE = "/home/asm/LHC-AD/Working files/Low_level_features.h5"
    OUTPUT_FILE = "/home/asm/LHC-AD/Working files/3C_jet_images_parallel.h5"
    
    preprocess(INPUT_FILE, OUTPUT_FILE)
