function [Y] = spectral_clustering(Affinity, num_classes)
%  [Y] = spectral_clustering(L, chosen_eig_indices, num_classes)
%      a skeleton function to perform spectral clustering, needs to be completed
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

L =  build_laplacian(Affinity, 'unn');

if nargin < 2
    num_classes = 2;
end

[U,E] = eig(L);

[eigenvalues_sorted,reorder] = sort(diag(E));

U = U(:,reorder(num_classes));

Y = kmeans(U, num_classes);
