import torch


def knn_dot_product(A: torch.Tensor, B: torch.Tensor, k: int):
    dot_products = A @ B.T  # Compute the dot products
    return torch.argsort(dot_products, dim=1, descending=True)[:, :k]
