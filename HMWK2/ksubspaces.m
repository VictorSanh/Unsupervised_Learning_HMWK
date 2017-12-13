function [global_groups, global_obj] = ksubspaces(data, n, d, replicates, ground_truth)
    % data: D by N data matrix.
    % n: number of subspaces
    % d: dimension of subspaces
    % replicates: number of restarts

    [D,N] = size(data);
    max_iter = 1;
    groups_history = zeros(replicates, N);
    obj_history = {};
    err_history = zeros(replicates, 1);
    
    for e=1:replicates
        fprintf("Run %i\n", e);
        
        % Initialization
        x = randperm(size(data,2),n);
        means = data(:,x);
        y = {} ;
        U = {} ;
        for i = 1:n
            temp = eye(D) ;%RandOrthMat(D);
            U{i} = temp(:, 1:d{i}); 
        end

        % EM
        for k = 1:max_iter
            fprintf("iteration %i\n", k);
            %Segmentation
            fprintf("  Segmentation\n");
            distance = zeros(n, N);
            for l = 1:n
                tmp = (eye(D) - U{l}*U{l}')*(data -  means(:,l));
                distance(l,:) = diag(tmp'*tmp);
            end

            W = zeros(n, N);
            [~, idx] = min(distance);
            for j = 1:N
                W(idx(j), j) = 1;
            end

            %Estimation
            fprintf("  Estimation\n");
            for i = 1:n
                num = sum(repmat(W(i,:),D,1).*data, 2);
                dem = sum(W(i,:));
                means(:,i) = num/dem ;

                z = repmat(W(i,:), D, 1).*(data - means(:,i));
                [V, E] = eigs(z*z');
                [~, reorder] = sort(diag(E));
                U{i} = V(:,reorder(1:d{i})); 

                for j = 1:N
                    if W(i,j) == 1
                        y{j} = U{i}'*(data(:,j) - means(:,i));
                    end
                end
            end
        end
        
        [~ , groups_history(e, :)] = max(W);
       	obj_history{e} = {y, U};
        err_history(e) = clustering_error(groups_history(e, :), ground_truth);
    end

    
    [~, i] = min(err_history);
    global_groups = groups_history(i, :);
    global_obj = obj_history{i};
end