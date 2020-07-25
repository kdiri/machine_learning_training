function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
m = length(y); % number of training examples
n = length(theta);

updated_theta = theta;
updated_theta(1) = 0;
regularize_cost = sum(updated_theta.^2) * (lambda / (2 * m));
regularize_grad = updated_theta * (lambda / m);

J = (1 / (2 * m)) * sum(((X * theta) - y).^2) + regularize_cost;

% =========================================================================

% different behaviour for grad0 and for the rest.
grad = (1 / m) * X' * ((X * theta) - y) + regularize_grad;
grad = grad(:);

end
