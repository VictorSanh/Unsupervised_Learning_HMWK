clear all

% open data
d = dir('Hopkins155');
for i = 1:length(d)
	if ( (d(i).isdir == 1) && ~strcmp(d(i).name,'.') && ~strcmp(d(i).name,'..') )
		filepath = d(i).name;
		eval(['cd ' filepath]);

		f=dir;
		foundValidData = false;
		for j = 1:length(f)
			if ( ~isempty(strfind(f(j).name,'_truth.mat')) )
				ind = j;
				foundValidData = true;
				break
			end
		end
		eval(['load ' f(ind).name]);
		cd ..

		if (foundValidData)
			n = max(s);
			N = size(x,2);
			F = size(x,3);
			D = 2*F;
			X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);
        end
    end
end


% Clustering
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
fprintf('Error: %2.4f', error);