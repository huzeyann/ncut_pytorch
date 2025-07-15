# %%

from typing import List, Tuple
from ncut_pytorch import get_affinity, _plain_ncut, nystrom_ncut
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

def non_appriximation_ncut(features, num_eig, **config_kwargs):
    aff = get_affinity(features, **config_kwargs)
    eigvec, eigval = _plain_ncut(aff, num_eig)
    return eigvec, eigval

def run_ncut(ncut_fn, features, num_eig, config_kwargs={}):
    ret = ncut_fn(features, num_eig, **config_kwargs)
    eigvec, eigval = ret[0], ret[1]
    return eigvec, eigval


def gram_schmidt(matrix):
    """Orthogonalize a matrix column-wise using the Gram-Schmidt process.

    Args:
        matrix (torch.Tensor): A matrix to be orthogonalized (m x n).
            the second dimension is orthogonalized
    Returns:
        torch.Tensor: Orthogonalized matrix (m x n).
    """

    # Get the number of rows (m) and columns (n) of the input matrix
    m, n = matrix.shape

    # Create an empty matrix to store the orthogonalized columns
    orthogonal_matrix = torch.zeros((m, n), dtype=matrix.dtype)

    for i in range(n):
        # Start with the i-th column of the input matrix
        vec = matrix[:, i]

        for j in range(i):
            # Subtract the projection of vec onto the j-th orthogonal column
            proj = torch.dot(orthogonal_matrix[:, j], matrix[:, i]) / torch.dot(
                orthogonal_matrix[:, j], orthogonal_matrix[:, j]
            )
            vec = vec - proj * orthogonal_matrix[:, j]

        # Store the orthogonalized vector
        orthogonal_matrix[:, i] = vec / torch.norm(vec)

    return orthogonal_matrix


def correct_rotation(eigen_vector):
    # correct the random rotation (flipping sign) of eigenvectors
    rand_w = torch.ones(
        eigen_vector.shape[0], device=eigen_vector.device, dtype=eigen_vector.dtype
    )
    s = rand_w[None, :] @ eigen_vector
    s = s.sign()
    return eigen_vector * s


def post_process_eigvec(eigvec, correct_sign=True, renormalize=True):
    if correct_sign:
        eigvec = correct_rotation(eigvec)
    if renormalize:
        eigvec = gram_schmidt(eigvec)
    return eigvec


def check_eigvec_numerical(eigvec1, eigvec2, tol=1e-3, logger=None):
    
    check_passed = True
    logger = logger or logging.getLogger(__name__)
    
    # check the numerical difference between every num_eig
    n_nodes, n_eig = eigvec1.shape
    for i in range(n_eig):
        diff = eigvec1[:, i] - eigvec2[:, i]
        mean_diff = diff.abs().mean()
        norm_diff = diff.norm()
        max_diff = diff.abs().max()
        if mean_diff > tol:
            check_passed = False
            # logger.warning(f"eigvec {i} mean diff: {mean_diff:.2e}, norm diff: {norm_diff:.2e}, max diff: {max_diff:.2e}")
            logger.warning(f"eigvec {i} mean diff: {mean_diff:.2e}")
    return check_passed


def plot_two_eigvecs(eigvec1, eigvec2, x_2d, n_cols=5, vmaxmin=0.1, tol=1e-3):
    n_eig = eigvec1.shape[1]

    n_rows = n_eig // n_cols *2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))        
    # plot one row of eigvec1, one row of eigvec2, repeat
    for i in range(n_eig):
        mean_diff = (eigvec1[:, i] - eigvec2[:, i]).abs().mean()
        pass_check = mean_diff < tol
        
        ax = axs[i // n_cols * 2, i % n_cols]
        ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec1[:, i], cmap="bwr", vmin=-vmaxmin, vmax=vmaxmin)
        if not pass_check:
            ax.set_title(f"fn1 eig{i}\nmean diff: {mean_diff:.2e}", color="red")
        else:
            ax.set_title(f"fn1 eig{i}\nmean diff: {mean_diff:.2e}", color="green")
            
        ax = axs[i // n_cols * 2 + 1, i % n_cols]
        ax.scatter(x_2d[:, 0], x_2d[:, 1], c=eigvec2[:, i], cmap="bwr", vmin=-vmaxmin, vmax=vmaxmin)
        if not pass_check:
            ax.set_title(f"fn2 eig{i}\nmean diff: {mean_diff:.2e}", color="red")
        else:
            ax.set_title(f"fn2 eig{i}\nmean diff: {mean_diff:.2e}", color="green")
            
    return fig


if __name__ == "__main__":
    
    plot_save_dir = "./plots"

    import os
    os.makedirs(plot_save_dir, exist_ok=True)


    from sklearn.datasets import make_blobs
    x_2d, y = make_blobs(n_samples=1000, centers=10, n_features=2, random_state=42)
    x_2d = torch.tensor(x_2d, dtype=torch.float32)


    base_args = {
        "num_sample": 1000,
        "knn": 10,
        "sample_method": "farthest",
        "precomputed_sampled_indices": None,
        "distance": "cosine",
        "affinity_focal_gamma": 1.0,
        "indirect_connection": False,
        "indirect_pca_dim": 100,
        "device": None,
        "eig_solver": "svd_lowrank",
        "normalize_features": None,
        "matmul_chunk_size": 8096,
        "make_orthogonal": True,
        "verbose": False,
        "no_propagation": False,
    }

    logging.basicConfig(level=logging.INFO)

    num_eig = 10

    _i_test = 0
    # loop over different configurations
    for distance in ["cosine", "rbf", "euclidean"]:
        eigvec1, eigval1 = run_ncut(non_appriximation_ncut, x_2d, num_eig, {"distance": distance})
        eigvec1 = post_process_eigvec(eigvec1, correct_sign=True, renormalize=True)
        
        for num_sample in [1000, 100]:
            args = base_args.copy()
            args["distance"] = distance
            args["num_sample"] = num_sample
            
            eigvec2, eigval2 = run_ncut(nystrom_ncut, x_2d, 10, args)
            eigvec2 = post_process_eigvec(eigvec2, correct_sign=True, renormalize=True)
            
            logger = logging.getLogger(f"[distance={distance}][num_sample={num_sample}]")
            check_passed = check_eigvec_numerical(eigvec1, eigvec2, tol=1e-3, logger=logger)
            if not check_passed:
                logger.error("Check failed")
            else:
                logger.info("Check passed")

        
            fig = plot_two_eigvecs(eigvec1, eigvec2, x_2d, n_cols=5, vmaxmin=0.1, tol=1e-3)
            fig.suptitle(f"distance={distance}, num_sample={num_sample}")
            fig.tight_layout()
            fig.savefig(f"{plot_save_dir}/test_{_i_test}.png")
            plt.close(fig)
            _i_test += 1
            

# %%
