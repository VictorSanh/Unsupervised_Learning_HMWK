function W = build_affinity(X, sigma2)

k = floor(sqrt(size(X, 2)));

similarities = pdist2(X', X', 'euclidean');
similarities = exp(-similarities./(2*sigma2));

nb_points = size(X, 2);
W = zeros(nb_points, nb_points);

for i=1:nb_points
  %sorted_index_i is the indexes of the sorted line i of similarities
  [~, sorted_index_i] = sort(similarities(i,:), 'descend');  
  W(i, sorted_index_i(2:(k+1))) = similarities(i, sorted_index_i(2:(k+1)));
end

W = 1/2*(W+W');