function [global_groups, global_obj] = ksubspaces(data, n, d, replicates)
    % data: D by N data matrix.
    % n: number of subspaces
    % d: dimension of subspaces
    % replicates: number of restarts

    [D,N] = size(data);

    % Initialization
    x = randperm(size(data,2),n);
    means = data(:,x);
    y = {} ;

    U = {} ;

    for i = 1:n

        temp = eye(D) ;%RandOrthMat(D);
        U{i} = temp(:, 1:d{i}); 

    end

    max_iter = 2 ;

    for k = 1:max_iter

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
        
        for i = 1:n
            
            num = sum(repmat(W(i,:),D,1).*data, 2);
            dem = sum(W(i,:));
            means(:,i) = num/dem ;
            
            z = repmat(W(i,:), D, 1).*(data - means(:,i));
            [V, E] = eig(z*z');
            [eigen_values_sorted, reorder] = sort(diag(E));
            U{i} = V(:,reorder(1:d{i})); 
            
            for j = 1:N
                if W(i,j) == 1
                    y{j} = U{i}'*(data(:,j) - means(:,i));
                end
            end
            
                
        end


    end


    global_groups = 0;
    global_obj = 0;
end