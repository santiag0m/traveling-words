from sklearn.manifold import SpectralEmbedding
import umap


class UMAPWithLaplacianInit:
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=3):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.umap_model = None

    def fit(self, data):
        # Compute Laplacian Eigenmap embedding
        laplacian_embedding = SpectralEmbedding(
            n_components=self.n_components
        ).fit_transform(data)

        # Use Laplacian embedding as initialization for UMAP
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            init=laplacian_embedding,
        ).fit(data)

        return self.umap_model.embedding_

    def transform(self, new_data):
        if self.umap_model is None:
            raise ValueError("Model needs to be fitted before transforming data.")

        return self.umap_model.transform(new_data)
