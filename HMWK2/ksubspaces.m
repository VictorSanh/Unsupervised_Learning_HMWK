function [global_groups, global_obj] = ksubspaces(X, n, d, replicates, max_iter, ground_truth)
    % data: D by N data matrix.
    % n: number of subspaces
    % d: dimension of subspaces
    % replicates: number of restarts

    [D, N] = size(X);
    err = 1e5;
    
    for e=1:replicates
        fprintf("Run %i\n", e);
        
        % Initialization
        fprintf("  Initialization\n");
        selec = randperm(size(X, 2),n);
        means = X(:, selec);
        y = {} ;
        U = {} ;
        for i = 1:n
            U{i} = RandOrthMat(D, d{i}); %eye(D) ;
        end

        % EM
        fprintf("  Begin Alternate Minimization\n");
        %Convergence reached: clustering is equal between two successive
        %clustering
        previous_group = zeros(1, N);
        
        for k = 1:max_iter
            fprintf("  Iteration %i\n", k);
            
            %Segmentation
            fprintf("    Segmentation\n");
            distance = zeros(n, N);
            %distance(i, j) = distance of point xj to subspace Si
            %This step is time-consuming.... If optimization is possible
            %optimize here
            for i = 1:n
                tmp = (eye(D) - U{i}*U{i}')*(X -  means(:,i));
                distance(i,:) = diag(tmp'*tmp)';
            end
            
            
            W = zeros(n, N);
            [~, idx] = min(distance);
            for j = 1:N
                W(idx(j), j) = 1;
            end

            %Estimation
            fprintf("    Estimation\n");
            for i = 1:n
                num = sum(repmat(W(i,:), D, 1).*X, 2);
                dem = sum(W(i,:));
                means(:,i) = num/dem ;

                z = repmat(W(i,:), D, 1).*(X - means(:,i));
                [V, ~] = eigs(z*z', d{i});
                U{i} = V; 
            end
            for j = 1:N
                [~, i] = max(W(:, j));
                y{j} = U{i}'*(X(:,j) - means(:,i));
            end
            
            %Check for convergence
            [~, current_group] = max(W);
            if isequal(current_group, previous_group)
                fprintf("    Convergence reached in %i iterations\n", k);
                break;
            end
            previous_group = current_group;
        end
        
        
        group_k = current_group;
        current_error = clustering_error(group_k, ground_truth);
       	if (current_error < err)
            global_obj = {y, U};
            err = current_error;
            global_groups = group_k;
        end
            
    end
end