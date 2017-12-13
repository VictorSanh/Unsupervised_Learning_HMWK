function error = clustering_error(label, groups)
% label: N-dimensional vector with ground truth labels for a dataset with N points
% groups: N-dimensional vector with estimated labels for a dataset with N points

% permute labels of L2 to match L1 as good as possible
label = label(:);
groups = groups(:);

label1 = unique(label);
nclass1 = length(label);
label2 = unique(groups);
nclass2 = length(groups);

nclass = max(nclass1,nclass2);
G = zeros(nclass);
for i=1:nclass1
	for j=1:nclass2
		G(i,j) = length(find(label == label1(i) & groups == label2(j)));
	end
end

[c,t] = hungarian(-G);
grps = zeros(size(groups));
for i=1:nclass2
    grps(groups == groups(i)) = label(c(i));
end
error = sum(label ~= grps) / length(label);
end