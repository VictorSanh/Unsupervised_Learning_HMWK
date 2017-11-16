%% 2. Face Completion
%% 2. Face Completion

clear all
img = {};
W = {};
errors = {};
missing_entries = [0.2, 0.4, 0.6, 0.8];
taus = [5e3, 1e4, 1e5, 1e6];
figure(3); hold on
% Looping over tau
for k=1:4
    tau = taus(k);
    % Looping over the missing entries percentage
    for j=1:4
        errors{j}={};
        % Looping on the images
        for i=1:10
            %Loading image
            img{i} = loadimage(1,i);
            %Randomly selecting the observed entries
            W{i} = binornd(1, 1-missing_entries(j), 192, 168);

            %Using Low Rank Matrix Completion to evaluate the full matrix
            [A, errors{j}{i}] = lrmc(img{i}, W{i}, tau, 2);

            %Display the observed initial face
            figure(1);
            imshow(uint8(W{i}.*img{i}));

            %Display the retrieved face
            figure(2);
            imshow(uint8(A));
        end
        errors{j} = mean([errors{j}{:}]);
    end
    
figure(3), plot(missing_entries, [errors{:}])
end
hold off
legend()
xlabel('Percentage of missing entries')
ylabel('Mean MSE for 10 pictures')


%% 3. Movie Recommendation Grand Challenge.
% c'est faux -> il faut enlever des ratings.
%Probablement
clear all

movie = csvread('movies/ratings_medium_n4_Horror_Romance_42.csv',1,0);

p = .95;      % proportion of rows to select for training
N = size(movie,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;     
tf = tf(randperm(N));   % randomise order
dataTraining = movie(tf,:);
dataTesting = movie(~tf,:);

%remove randomly x% of ratings in the Training database
missing_entries = 0.5;
tau = 1e4;
N_train = size(dataTraining,1);

%Initialization
dataTraining_missing = dataTraining;

indexes_missing_entries = binornd(1, 1-missing_entries, N_train, 1);
dataTraining_missing(:,4,:) = dataTraining_missing(:,4,:).*(indexes_missing_entries>0); %rating on column 4