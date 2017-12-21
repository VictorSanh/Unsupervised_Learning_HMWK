clear all

load('data/ExtendedYaleB.mat')
X = EYALEB_DATA;
y = EYALEB_LABEL;

% Search for the best parameters
X_sub = X(:, (y==1 | y==2));
y_sub = y((y==1 | y==2));

num_classes = length(unique(y_sub));

algorithm = 'ssc';

if strcmp(algorithm, 'spectral_clustering')
    K = [num_classes, 3, 5, 10];
    W = build_affinity(X_sub, mean(var(X_sub)));
    list_error = [];
    % Looping over K
    for k=1:length(K)
        groups = spectral_clustering(W, K(k), 'rw');
        error = clustering_error(y_sub, groups);
        list_error(k) = error ;
        fprintf('Error: %2.4f\n', error);
    end
    [best_K, K_opt] = min(list_error);
    fprintf('The best replicate value is : %d\n', K(K_opt));
    figure();
    plot(K, list_error);    

elseif strcmp(algorithm, 'k-subspaces')
    replicates = [5,10,20,50];
    max_iter = 50;
    d = repmat({3}, num_classes, 1);
    list_error = []
    % Looping over replicate values
    for r=1:length(replicates)
        [groups, obj] = ksubspaces(X_sub, num_classes, d, replicates(r), max_iter, y_sub);
        error = clustering_error(y_sub, groups);
        fprintf('Error: %2.4f\n', error);
        list_error(r) = error ;
    end
    [best_error, replicates_opt] = min(list_error);
    fprintf('The best replicate value is : %d\n', replicates(replicates_opt));
    figure(1);
    plot(replicates, list_error);
    
elseif strcmp(algorithm, 'ssc')
    taus = [1, 10, 20, 50, 100];
    mu2 = 800;
    list_error = [];
    % Looping over tau
    for k=1:length(taus)
        groups = SSC(X_sub, num_classes, taus(k), mu2);
        error = clustering_error(y_sub, groups);
        list_error(k) = error ;
        fprintf('Error: %2.4f\n', error);
    end
    [best_error, tau_opt] = min(list_error);
    fprintf('The best Tau value is : %d\n', taus(tau_opt));
    figure();
    plot(taus, list_error); 
    
    mu2s = [100, 200, 500, 800, 1000];
    list_error = [];
    % Looping over mu2
    for k=1:length(mu2s)
        groups = SSC(X_sub, num_classes, taus(tau_opt), mu2s(k));
        error = clustering_error(y_sub, groups);
        list_error(k) = error ;
        fprintf('Error: %2.4f\n', error);
    end
    [best_error, mu2_opt] = min(list_error);
    fprintf('The best mu2 value is : %d\n', mu2s(mu2_opt));
    figure();
    plot(taus, list_error); 
    
    tau = taus(tau_opt);
    mu2 = mu2s(mu2_opt);
    groups = SSC(X_sub, num_classes, tau, mu2);
end

% error
% error = clustering_error(y_sub, groups);
% fprintf('Error: %2.4f\n', error);  
