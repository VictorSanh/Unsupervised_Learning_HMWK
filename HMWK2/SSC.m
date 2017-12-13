function groups = SSC(data, n, tau, mu2)
% data: D by N data matrix.
% n: number of clusters
% tau, mu2: parameter

% sparse optimization program
[C] = lasso_min(data, mu2, tau);
% affinity matrix
W = abs(C) + abs(C)';
% normalized spectral clustering
[groups] = spectral_clustering(W, n, 'sym');
end