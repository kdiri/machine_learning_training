function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
c_values = [0.01 0.03 0.1 0.3 1.0 3.0 10.0 30.0];
m = length(c_values);
minima = error = 100;
c_val = sigma_val = 0 ;

for i = 1:m
    for j = 1:m
        training_model = svmTrain(X, y, c_values(i), @(x1, x2) gaussianKernel(x1, x2, c_values(j)));
        predictions = svmPredict(training_model, Xval);
        error = mean(double(predictions ~= yval));
        if(error < minima)
            minima = error;
            c_val = c_values(i);
            sigma_val = c_values(j);
        end
    end
end

C = c_val;
sigma = sigma_val;


% =========================================================================

end
