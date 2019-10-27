function [C, sigma] = dataset3Params2(X, y, Xval, yval)
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

% Each combination of sigma and C produces a single vector of predictions.
% This vector is then compared with the actual y values to give a single error
% percentage.
% The error values for all combinations are stored in a 2D matrix err.

C_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
err = zeros(length(C_test),length(sigma_test));

for i = 1:length(C_test)
    for j = 1:length(sigma_test)
        model= svmTrain(X, y, C_test(i), @(x1, x2) gaussianKernel(x1, x2, sigma_test(j)));
        pred = svmPredict(model, Xval);
        err(i,j) = mean(double(pred ~= yval));
    end
end

% Turn 2D error matrix into a single column.
[~,idx] = min(err(:)); % Returns minimum value (not interested) and its index.
[row,col] = ind2sub(size(err),idx); % Convert to row-column index from single column index.
C = C_test(row);
sigma = sigma_test(col);

% =========================================================================

end
