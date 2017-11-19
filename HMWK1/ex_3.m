clear all

%Load Movielens table
movie = readtable('movies/ratings_medium_n4_Horror_Romance_42.csv');

%Select only userId, movie Id and movie ratings
movieSub = table(movie.userId, movie.movieId, movie.rating);
movieSub.Properties.VariableNames = {'userId', 'movieId', 'rating'};

%Cast the data (transform line into column)
casted = unstack(movieSub, 'rating', 'movieId');
casted = table2array(casted);

%Using the names of lrmc
X = casted(:, 2:end); %first column was userId, we don't need it
W = 1-isnan(X);
X(isnan(X)) = 0;



tau = 1e4;
beta = 2;

[A, mse] = lrmc(X, W, tau, beta);


% TO DO:
% - Threshold the ratings obtained (now the ratings completed are real values.
% Sometimes it is negative, sometimes, it is higher than 5). Negative values should
% be corrected to a 0 rating. Higher than 5 ratings should be corrected to
% 5 rating.
% - (Optional) Threshold the obtained ratings to have a discrete range between 
% 0 and 5 (initial ratings can be 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5).
% - Sample the matrix to make it more sparse (80% training and 20% test)
% - Create a measurement of the mean squared error between test and ground truth.
% - Test it with only horror class
% - Test it with only romance class
% - Test it with horror+romance class