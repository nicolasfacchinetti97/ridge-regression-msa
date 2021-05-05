import numpy as np

class PCA:
    def __init__(self, data):
        self.data = data
        cov_mat = np.cov(data, rowvar=False)                                # get covariance matrix
        eigen_values , eigen_vectors = np.linalg.eig(cov_mat)          # get eigen values and eigen vectors 
        
        sorted_index = np.argsort(eigen_values)[::-1]                   # sort the index of eigenvalues in descending order
        self.eigenvalues = eigen_values[sorted_index]                  
        self.eigenvectors = eigen_vectors[:,sorted_index]

    def get_principal_components(self, n_components):
        eigenvector_subset = self.eigenvectors[:,0:n_components]
        X_reduced = np.dot(eigenvector_subset.transpose(), self.data.transpose()).transpose()
        return X_reduced
     
    def get_percentage_of_variance(self):
        total_variance = np.sum(self.eigenvalues)

        variances = []
        for value in self.eigenvalues:
            variances.append(value/total_variance)
        return variances