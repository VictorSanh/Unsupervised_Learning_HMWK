function [C1] = lasso_min(data, mu2, tau)
N = size(data,2);
% initialization
A = inv(tau*(data'*data)+mu2*eye(N));
C1 = zeros(N,N);
Lambda2 = zeros(N,N);
err = 2*10^-3; 
i = 1;
while ( err(i) > thr1 && i < maxIter )
    % updating Z
    Z = A * (tau*(data'*data)+mu2*(C1-Lambda2/mu2));
    Z = Z - diag(diag(Z));
    % updating C2
    C2 = max(0,(abs(Z+Lambda2/mu2) - 1/mu2*ones(N))) .* sign(Z+Lambda2/mu2);
    C2 = C2 - diag(diag(C2));
    % updating Lambda
    Lambda2 = Lambda2 + mu2 * (Z - C2);
    % computing errors
    err(i+1) = max(max(abs(Z-C2)));

    C1 = C2;
    i = i + 1;
end
end