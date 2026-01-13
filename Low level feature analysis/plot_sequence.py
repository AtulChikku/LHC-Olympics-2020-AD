import matplotlib.pyplot as plt
import numpy as np

def plot_sequence(jet_coords, jet_pt, order, title=None):
    """
    Visualize the ordering of pixels in eta-phi space.

    Parameters
    ----------
    jet_coords : torch.Tensor, shape (N, 2)
        (eta, phi) coordinates of active pixels
    jet_pt : torch.Tensor, shape (N,)
        pT values for each pixel
    order : list or 1D array of length N
        Ordering of pixel indices (from C/A or kT-inspired)
    title : str, optional
        Plot title
    """

    # Convert to numpy for plotting
    eta = jet_coords[:, 0].detach().cpu().numpy()
    phi = jet_coords[:, 1].detach().cpu().numpy()
    pt  = jet_pt.detach().cpu().numpy()

    order = np.array(order)

    plt.figure(figsize=(5, 5))

    # Plot all active pixels (background)
    plt.scatter(
        eta, phi,
        c=pt,
        cmap="viridis",
        s=12,
        alpha=0.35,
        label="All pixels"
    )

    # Plot sequence path
    plt.plot(
        eta[order],
        phi[order],
        "-o",
        color="red",
        linewidth=1,
        markersize=3,
        label="Sequence"
    )

    # Highlight starting point
    plt.scatter(
        eta[order[0]],
        phi[order[0]],
        s=80,
        facecolors="white",
        edgecolors="black",
        zorder=5,
        label="Start"
    )

    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\phi$")
    if title is not None:
        plt.title(title)

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
