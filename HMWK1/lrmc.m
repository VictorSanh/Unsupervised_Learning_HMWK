function [A, mse] = lrmc(X, W, tau, beta)
    %INPUTS
    %X D*N data matrix
    %W D*N binary matrix doneting knwown (1) or missing (0) entries
    %tau Parameter of he optimization problem
    %beta Step size of the dual gradient ascent step
    
    %OUTPUTS
    % A Low-rank completion of the matrix X
    
    
    %Other Parameters
    max_iter = 5000;
    eps = 1e-2;
    [D, N] = size(X);
    
    %Initialization
    Z_current = zeros(D, N);
    %Z_current = beta*W.*X;    

    for k=1:max_iter
        [U, S, V] = svd(W.*Z_current);
        A_next = U*singular_value_threshold(S, tau)*V';
        Z_next = Z_current + beta*(W.*(X-A_next));
        
        %Check for convergence
        if norm(W.*(X - A_next), 'fro')/norm(W.*X, 'fro') < eps
            break
        end
        %If not converged, proceed to the next iteration
        Z_current = Z_next;
    end
    
    A = A_next;
    fprintf("Convergence reached in %i iterations\n", k);
    mse = immse(W.*X, W.*A_next);
    fprintf("Training MSE : %0.5f\n", mse);
end
