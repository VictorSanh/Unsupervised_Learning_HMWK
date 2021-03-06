function [L] =  build_laplacian(Affinity, laplacian_normalization)
%  [L] =  build_laplacian(X, graph_param, laplacian_normalization)
%
%  Input
%  X:
%      Affinity: N by N affinity matrix, where N is the number of points.
%  laplacian_normalization:
%      string selecting which version of the laplacian matrix to construct
%      either 'unn'normalized, 'sym'metric normalization
%      or 'rw' random-walk normalization
%
%  Output
%  Y:
%      Cluster assignments

W = Affinity;

D = diag(sum(W));

if strcmp(laplacian_normalization, 'unn')
    L = D - W;
elseif strcmp(laplacian_normalization, 'sym')
    L = inv(D)^(1/2)*(D - W)*inv(D)^(1/2);
elseif strcmp(laplacian_normalization, 'rw')
    L = eye(size(W)) - inv(D)*W;
else
    error('unkown normalization mode')
end

