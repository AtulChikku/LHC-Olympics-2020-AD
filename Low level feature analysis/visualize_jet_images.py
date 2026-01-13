import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_H5 = "/home/asm/LHC-AD/Working files/3C_jet_images_parallel.h5"
FEATURE_NAMES = ['Normalized pT', 'Normalized E', 'Log(Normalized pT)']
N_FEATURES = 3

def visualize_jet_images(event_indices):
    """
    Loads jet images from the HDF5 file and displays the three feature channels.
    
    Args:
        event_indices (list[int]): A list of integer indices (row numbers) 
                                   to load from the HDF5 file.
    """
    try:
        with h5py.File(INPUT_H5, "r") as f:
            images = f["images"]
            mjj = f["MJJ"]
            labels = f["labels"] # Assuming you added the labels dataset

            for idx in event_indices:
                if idx >= len(images):
                    print(f"Index {idx} is out of bounds (max {len(images) - 1}). Skipping.")
                    continue

                img = images[idx] # Shape (3, 50, 50)
                mjj_val = mjj[idx]
                label_val = labels[idx]
                
                # Check for skipped events (MJJ=-1.0 sentinel value)
                if mjj_val == -1.0:
                    print(f"Index {idx}: Event was skipped during preprocessing (MJJ = -1.0).")
                    continue
                
                # --- Plotting ---
                fig, axes = plt.subplots(1, N_FEATURES, figsize=(15, 5))
                fig.suptitle(
                    f"Event Index: {idx} | Label: {label_val} | MJJ: {mjj_val:.2f} GeV",
                    fontsize=14
                )
                
                for i in range(N_FEATURES):
                    ax = axes[i]
                    data = img[i]
                    
                    # Use 'viridis' for pT/E and 'inferno' or 'magma' for log-pT, 
                    # but 'viridis' is a safe default.
                    im = ax.imshow(data, cmap='viridis', origin='lower')
                    
                    ax.set_title(FEATURE_NAMES[i])
                    ax.set_xlabel(r'$\Delta\phi$ bins')
                    ax.set_ylabel(r'$\Delta\eta$ bins')
                    
                    # Add a color bar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_H5}. Please check the path.")
    except KeyError as e:
        print(f"Error: Dataset {e} not found in the HDF5 file. Ensure dataset names are 'images', 'MJJ', and 'labels'.")

# ----------------------------------------------------
# EXECUTION
# ----------------------------------------------------
if __name__ == "__main__":
    
    # 1. Background Event Indices (First 1 million entries)
    bg_indices = [123932]
    
    # 2. Signal Event Indices (Last 100k entries, assuming 1.1 million total)
    # Use indices near the boundary (e.g., 1,000,000 + 100)
    # Check your total dataset size if 1.1 million is the true total.
    # Assuming N_total = 1,100,000
    signal_start = 1_000_000 
    signal_indices = [1000035]
    
    # Combine lists and visualize
    test_indices = bg_indices + signal_indices
    visualize_jet_images(test_indices)