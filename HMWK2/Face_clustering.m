clear all

load('data/ExtendedYaleB.mat')
X = EYALEB_DATA;
y = EYALEB_LABEL;

num_classes = length(unique(y));



algorithm = 'k-subspaces';

if strcmp(algorithm, 'spectral_clustering')
    W = build_affinity(X, mean(var(X)));
    groups = spectral_clustering(W, num_classes, 'rw');
    

elseif strcmp(algorithm, 'k-subspaces')
    replicates = 2;
    d = {1,2,3};
    [groups, obj] = ksubspaces(X, 3, d, replicates);

elseif strcmp(algorithm, 'ssc')
    tau = 20;
    mu2 = 800;
    groups = SSC(X, num_classes, tau, mu2);
end

% error
error = clustering_error(y, groups);
fprintf('Error: %2.4f\n', error);
