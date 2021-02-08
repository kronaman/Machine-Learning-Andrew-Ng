function [C, sigma, error_vals] = dataset3Params(X, y, Xval, yval)
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
error = 0;
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

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
min_error = mean(predictions ~= yval);


C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_vals = zeros(length(C_vals), length(sigma_vals));

for i = 1:length(C_vals)
    for j = 1:length(sigma_vals)
        model= svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j)));
        predictions = svmPredict(model, Xval);
        error = mean(predictions ~= yval);
        if (error < min_error)
            C = C_vals(i);
            sigma = sigma_vals(j);
            min_error = error;
        end
        error_vals(i, j) = error;
    end
end


% =========================================================================

end
