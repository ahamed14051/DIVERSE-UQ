import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

class DiversityCalculator:
    @staticmethod
    def mean_pairwise_distance(features, metric='cosine'):
        dists = pdist(features, metric=metric)
        return np.mean(dists)
    
    @staticmethod
    def variance_score(features):
        return np.trace(np.cov(features.T))
    
    # @staticmethod
    # def entropy_score(features, n_clusters=5):
    #     kmeans = KMeans(n_clusters=n_clusters).fit(features)
    #     counts = np.bincount(kmeans.labels_)
    #     probs = counts / np.sum(counts)
    #     return -np.sum(probs * np.log(probs + 1e-9))

    @staticmethod
    def entropy_score(features, n_clusters=2):  # Reduced number of clusters
        kmeans = KMeans(n_clusters=n_clusters).fit(features)
        counts = np.bincount(kmeans.labels_)
        probs = counts / np.sum(counts)
        return -np.sum(probs * np.log(probs + 1e-9))

    
    @classmethod
    def compute_all_scores(cls, features):
        return {
            'mean_pairwise': cls.mean_pairwise_distance(features),
            'variance': cls.variance_score(features),
            'entropy': cls.entropy_score(features)
        }