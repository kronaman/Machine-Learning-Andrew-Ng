function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta) - 1;

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_x = sigmoid(X * theta);
 
J = (-1 / m) * (y' * log(h_x) + (1 - y)' * log(1 - h_x)) + (lambda / (2 * m)) * (theta' * theta - theta(1) * theta(1));
% Regularizing all parameters except theta0
% J = J - theta(1) * theta(1);

grad = (1/m) * (X' * (h_x - y)) + (lambda / m) * theta;
grad(1) = grad(1) - (lambda / m) * theta(1);


% =============================================================

end
