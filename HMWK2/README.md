## Project 2 – Unsupervised Learning

### Clustering Algorithms

We have implemented three clustering algorithms : Spectral Clustering algorithm (using MATLAB kmeans), 
K-Subspaces algorithm and Sparse Subspace Clustering (SSC) algorithm with noisy data.  
We have also implemented the clustering error using the Hungarian algorithm (to efficiently 
to evaluate the error for all possible permutation)

### Application 1 : Face Clustering

We have used the ExtendedYaleB dataset to cluster faces (face images of 38 subjects, 
each under 64 different illumination conditions) and we have applied the three algorithms on this dataset.


### Application 2 : Motion Segmentation

We have used the Hopkins155 dataset to cluster feature point trajectories and we have applied 
the three algorithms on this dataset.

### Files
- **build_laplacian.m** : function to construct a laplacian from affinity matrix  
- **clustering_error.m** : function to compute the clustering error using Hungarian algorithm  
- **Face_clustering.m** : Application 1  
- **hungarian.m** : Hungarian implementation from Niclas Borlin  
- **ksubspaces.m** : function for the K-Subspaces algorithm  
- **lasso_min.m** : function for Matrix LASSO Minimization by ADMM  
- **Motion_Segmentation.m** : Application 2  
- **RandOrthMat.m** : function to generate a random n x n orthogonal real matrix  
- **spectral_clustering.m** : function to perform spectral clustering  
- **SSC.m** : function to perform Sparse Subspace Clustering
