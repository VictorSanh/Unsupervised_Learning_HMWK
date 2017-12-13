clear all

load('data/ExtendedYaleB.mat')
X = EYALEB_DATA;
y = EYALEB_LABEL;

num_classes = length(unique(y));

%%%%%
%Building a graph (knn)
%%%%%
sigma2 = 3e3;

similarities = pdist2(X', X', 'euclidean');
similarities = sqrt(similarities);
similarities = exp(-similarities./(2*sigma2));

nb_points = size(y, 2);
W = zeros(nb_points, nb_points);

for i=1:nb_points
  %sorted_index_i is the indexes of the sorted line i of similarities
  [sorted_line_i, sorted_index_i] = sort(similarities(i,:));  
  W(i, sorted_index_i(1:num_classes)) = similarities(i, sorted_index_i(1:num_classes));
end
W = 1/2*(W+W');

%%%%%
%Spectral clustering
%%%%%
out = spectral_clustering(W, num_classes);

