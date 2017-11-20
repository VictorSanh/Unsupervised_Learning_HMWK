clear all

%Load Movielens table
movie = readtable('movies/ratings_medium_n4_Horror_Romance_42.csv');
N = size(movie,1);

%Select only userId, movie Id and movie ratings
movieSub = table(movie.userId, movie.movieId, movie.rating);
movieSub.Properties.VariableNames = {'userId', 'movieId', 'rating'};

%Remove some ratings to make the matrix more sparse
missing_entries = 0.2 ;
indexes_missing_entries = 1 - binornd(1, 1-missing_entries, N, 1);
movieSub_MissingRatings = movieSub ;
movieSub_MissingRatings(logical(indexes_missing_entries),3) = {0} ;

%Cast the data (transform line into column)
casted = unstack(movieSub, 'rating', 'movieId');
casted = table2array(casted);
casted_missing = unstack(movieSub_MissingRatings, 'rating', 'movieId');
casted_missing = table2array(casted_missing);

%Using the names of lrmc
X = casted(:, 2:end); %first column was userId, we don't need it
X_missing = casted_missing(:, 2:end);
W = 1-isnan(X_missing);
X(isnan(X)) = 0;


tau = 1e4;
beta = 2;

[A, mse] = lrmc(X, W, tau, beta);

%Threshold the ratings obtained
A(A>5) = 5 ;
A(A<0) = 0 ;

%Mean Square Error between test and ground truth
[x_test, y_test] = find(X_missing==0);
N_testdata = size(x_test,1);

GroundTruth = [];
for i=1:N_testdata
    GroundTruth = [GroundTruth X(x_test(i), y_test(i))];
end

Test = [];
for i=1:N_testdata
    Test = [Test A(x_test(i), y_test(i))];
end

MseTest = immse(GroundTruth,Test);
fprintf("MSE GroundTruth/Test: %0.2f\n", MseTest);

% TO DO:
% - Threshold the ratings obtained (now the ratings completed are real values.
% Sometimes it is negative, sometimes, it is higher than 5). Negative values should
% be corrected to a 0 rating. Higher than 5 ratings should be corrected to
% 5 rating. -> DONE
% - (Optional) Threshold the obtained ratings to have a discrete range between 
% 0 and 5 (initial ratings can be 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5).
% - Sample the matrix to make it more sparse (80% training and 20% test) ->
% DONE
% - Create a measurement of the mean squared error between test and ground
% truth. -> DONE
% - Test it with only horror class
% - Test it with only romance class
% - Test it with horror+romance class