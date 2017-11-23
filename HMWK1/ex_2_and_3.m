% 2. Face Completion

clear all
img = {};
W = {};
errors = {};
missing_entries = [0.2, 0.4, 0.6, 0.8];
taus = [5e3, 1e4, 1e5, 1e6, 1e7];
figure(3); hold on
% Looping over tau
for k=1:5
    tau = taus(k);
    % Looping over the missing entries percentage
    for j=1:4
        errors{j}={};
        mse{j} = {};
        % Looping on the images
        for i=1:10
            %Loading image
            img{i} = loadimage(1,i);
            %Randomly selecting the observed entries
            W{i} = binornd(1, 1-missing_entries(j), 192, 168);

            %Using Low Rank Matrix Completion to evaluate the full matrix
            [A, errors{j}{i}] = lrmc(img{i}, W{i}, tau, 2);
            mse{j}{i} = immse(img{i},A); 

            %Display the observed initial face
            figure(1);
            imshow(uint8(W{i}.*img{i}));

            %Display the retrieved face
            figure(2);
            imshow(uint8(A));
        end
        errors{j} = mean([errors{j}{:}]);
        mse{j} = mean([mse{j}{:}]);
    end
    
figure(3), plot(missing_entries, [mse{:}])
end
hold off
legend('tau = 5e3', 'tau = 1e4', 'tau = 1e5', 'tau = 1e6', 'tau = 1e7')
xlabel('Percentage of missing entries')
ylabel('Mean MSE for 10 pictures')