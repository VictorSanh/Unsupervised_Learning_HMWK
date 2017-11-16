%% Exo 2

clear all

img = {};
W = {};
for i=1:1
    %Loading images
    img{i} = loadimage(1,i);
    %Randomly selecting the observed entries
    W{i} = binornd(1, 0.4, 192, 168);
    
    %Using Low Rank Matrix Completion to evaluate the full matrix
    A = lrmc(img{i}, W{i}, 1e4, 2);
    
    %Display the observed initial face
    figure(1);
    imshow(uint8(W{i}.*img{i}));
    
    %Display the retrieved face
    figure(2);
    imshow(uint8(A));
end




%% Exo3
% c'est faux -> il faut enlever des ratings.
%Probablement
clear all

movie = csvread('movies/ratings_medium_n4_Horror_Romance_42.csv', 1, 0);

p = .95;      % proportion of rows to select for training
N = size(movie,1);  % total number of rows 
tf = false(N,1);    % create logical index vector
tf(1:round(p*N)) = true;     
tf = tf(randperm(N));   % randomise order
dataTraining = movie(tf,:);
dataTesting = movie(~tf,:);

