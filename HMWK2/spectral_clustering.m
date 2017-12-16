function [Y] = spectral_clustering(Affinity, num_classes, laplacian_normalization)
%  [Y] = spectral_clustering(L, chosen_eig_indices, num_classes)
%      a skeleton function to perform spectral clustering
%
%  Input
%  L:
%      Graph Laplacian (standard or normalized)
%  num_classes:
%      number of clusters to compute (defaults to 2)
%
%  Output
%  Y:
%      Cluster assignments
if nargin < 3
    laplacian_normalization = 'unn';
end

L =  build_laplacian(Affinity, laplacian_normalization);

if nargin < 2
    num_classes = 2;
end

[U,E] = eig(L);

[~, reorder] = sort(diag(E));

U = U(:,reorder(1:num_classes));

Y = kmeans(U, num_classes, 'replicates', 15);
