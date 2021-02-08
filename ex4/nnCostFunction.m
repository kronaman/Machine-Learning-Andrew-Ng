function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m, 1) X];
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));

for i = 1:m

    %     Forward propagation:
    z_2 = Theta1 * X(i, :)';
    a_2 = sigmoid(z_2);
    a_2 = [1;a_2];
    a_3 = sigmoid(Theta2 * a_2); % vector of (num_labels, 1) dimension

    %     cost without regularization
    y_k = 1:num_labels ; % row vector of (num_labels, 1) dimension
    y_k = y_k == y(i,1);
    J = J - (y_k * log(a_3) + (1 - y_k) * log(1 - a_3));

    %     back propagation:
    del_3 = zeros(num_labels, 1);
    del_2 = zeros(length(a_2), 1);
    
    
    del_3 = a_3 - y_k';
    del_2 = (Theta2' * del_3) .* a_2 .* (1 - a_2);

    % unregularized gradient term summed over all training examples in
    % delta_2 and delta_1
    delta_2 = delta_2 + (del_3 * a_2');
    delta_1 = delta_1 + (del_2(2:end) * X(i, :)); % a_1' = X(i,:) (26, 1) * (1, 401) = (26, 401) -> drop the first row
end

J = J / m;
Theta1_grad = (1.0 / m) * (delta_1);
Theta2_grad = (1.0 / m) * (delta_2);

% ADD THE regularization part to the cost and gradient
if lambda ~= 0
    J = J + ((lambda / (2.0 * m)) * (sumsqr(Theta1(:,2:end)) + sumsqr(Theta2(:,2:end))));
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * (Theta1(:, 2:end));
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * (Theta2(:, 2:end));
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
