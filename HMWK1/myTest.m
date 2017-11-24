clear all

missing_entries = 0.2;

%Load Movielens table
movie = readtable('movies/ratings_medium_n4_Horror_Romance_42.csv');


%Subset a of movies
%movie_class = "all"; %(1+2)
movie_class = "horror"; %(1)
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

%%


taus = [1e3, 1e4, 1e5, 1e6];
betas = [1.5, 2, 2.5];

for k=1:4
    tau = taus(k);
    for j=1:3
        beta = betas(j);
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
        fprintf("%s - tau %0.1f - beta %0.1f - MSE GroundTruth/Test: %0.2f\n",movie_class, tau, beta, MseTest);

    end
end