clear all


tau = 1e5;
beta = 2.5;
missing_entries = 0.2 ;


%Load Movielens table
movie = readtable('movies/ratings_medium_n4_Horror_Romance_42.csv');


%Subset a of movies
movie_class = "all"; %(1+2)
%movie_class = "horror"; %(1)
%movie_class = "romance"; %(2)

if movie_class=="romance"
    rows = movie.genreId==2;
    movie = movie(movie.genreId==2, :);
elseif movie_class=="horror"
    movie = movie(movie.genreId==1, :);
end


%Select only userId, movie Id and movie ratings
movie = table(movie.userId, movie.movieId, movie.rating);
movie.Properties.VariableNames = {'userId', 'movieId', 'rating'};


%Remove some ratings to build the test ratings set
N = size(movie, 1);
indexes_missing_entries = 1 - binornd(1, 1-missing_entries, N, 1);
movie_MissingRatings = movie ;
movie_MissingRatings(logical(indexes_missing_entries),3) = {0} ;


%Cast the data (transform line into column)
casted = unstack(movie, 'rating', 'movieId');
casted = table2array(casted);

casted_missing = unstack(movie_MissingRatings, 'rating', 'movieId');
casted_missing = table2array(casted_missing);


%Using the names of lrmc
X = casted(:, 2:end); %first column was userId, we don't need it
X_missing = casted_missing(:, 2:end);
[x_test, y_test] = find(X_missing==0);%Coordiantes of removed entries.

X_missing(isnan(X_missing)) = 0;
W = (X_missing>0);%Matrix of localization of known entries


%Call Matrix Completion
[A, training_mse] = lrmc(X_missing, W, tau, beta);


%Threshold the ratings obtained
A(A>5) = 5 ;
A(A<0.5) = 0.5 ;
A = round(2*A)./2;


%Mean Square Error estimated ratings and user ratings on the test ratings
N_testdata = size(x_test,1);

GroundTruth = zeros(1, N_testdata);%Vector of test user ratings
Estimated = zeros(1, N_testdata);%Vector of estimated user ratings
for i=1:N_testdata
    GroundTruth(i) = X(x_test(i), y_test(i));
    Estimated(i) = A(x_test(i), y_test(i));
end


MseTest = immse(GroundTruth, Estimated);
fprintf("%s - MSE GroundTruth/Test: %0.2f\n", movie_class, MseTest);

hist(Estimated - GroundTruth);



% TO DO:
% - Tune the parameters (tau and beta)


% DONE:
% - Threshold the ratings obtained (now the ratings completed are real values.
% Sometimes it is negative, sometimes, it is higher than 5). Negative values should
% be corrected to a 0.5 rating. Higher than 5 ratings should be corrected to
% 5 rating. -> DONE
% - (Optional) Threshold the obtained ratings to have a discrete range between 
% 0.5 and 5 (initial ratings can be 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)
% -> DONE
% - Sample the matrix to make it more sparse (80% training and 20% test) ->
% DONE
% - Create a measurement of the mean squared error between test and ground
% truth. -> DONE
% - Test it with only horror class (1) -> DONE
% - Test it with only romance class (2) -> DONE
% - Test it with horror+romance class -> DONE


% RESULTS up to now (format: missing_entries proportion, tau, beta, movie_class, MSE)
% 0.1 ; 1e5 ; 2 ; romance ; 1.67
% 0.2 ; 1e5 ; 1.5 ; romance ; 1.67
% 0.2 ; 1e3 ; 2 ; romance ; 2.25
% 0.2 ; 1e4 ; 2 ; romance ; 1.67
% 0.2 ; 1e5 ; 2 ; romance ; 1.45
% 0.2 ; 1e6 ; 2 ; romance ; 5.01
% 0.2 ; 1e5 ; 2.5 ; horror ; 1.56

% 0.1 ; 1e3 ; 2 ; all ; 2.07
% 0.1 ; 1e4 ; 2 ; all ; 1.62
% 0.1 ; 1e5 ; 2 ; all ; 1.56
% 0.2 ; 1e5 ; 1.5 ; all ; 1.65
% 0.2 ; 1e3 ; 2 ; all ; 2.29
% 0.2 ; 1e4 ; 2 ; all ; 1.64
% 0.2 ; 1e5 ; 2 ; all ; 1.63
% 0.2 ; 1e6 ; 2 ; all ; 4.46
% 0.2 ; 1e5 ; 2.5 ; all ; 1.51
% 0.2 ; 1e5 ; 3 ; all ; 1.58

% 0.1 ; 1e4 ; 2; horror ; 1.77
% 0.2 ; 1e3 ; 2 ; horror ; 2.37
% 0.2 ; 1e4 ; 2 ; horror ; 1.76
% 0.2 ; 1e4 ; 6 ; horror ; 1.88
% 0.2 ; 1e5 ; 2 ; horror ; 2.02
% 0.2 ; 1e6 ; 2 ; horror ; 5.71
% 0.2 ; 1e5 ; 2.5 ; horror ; 1.78

