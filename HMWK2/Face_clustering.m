clear all

load('data/ExtendedYaleB.mat')
X = EYALEB_DATA;
y = EYALEB_LABEL;

num_classes = length(unique(y));

%%%%%
%Building a graph (knn)
%%%%%
sigma2 = 3e3;
k = floor(sqrt(length(y))); % k of k-nn

similarities = pdist2(X', X', 'euclidean');
similarities = sqrt(similarities);
similarities = exp(-similarities./(2*sigma2));

nb_points = size(y, 2);
W = zeros(nb_points, nb_points);

for i=1:nb_points
  %sorted_index_i is the indexes of the sorted line i of similarities
  [sorted_line_i, sorted_index_i] = sort(similarities(i,:));  
  W(i, sorted_index_i(1:k)) = similarities(i, sorted_index_i(1:k));
end
W = 1/2*(W+W');

algorithm = 'ssc';

if strcmp(algorithm, 'spectral_clustering')
    groups = spectral_clustering(W, num_classes);

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
