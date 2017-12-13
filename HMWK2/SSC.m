function groups = SSC(data, n, tau, mu2)
% data: D by N data matrix.
% n: number of clusters
% tau, mu2: parameter

[C] = lasso_min(data, mu2, tau);
W = C + C';
[Y] = spectral_clustering(W, n, 'sym');
groups = Y;
end