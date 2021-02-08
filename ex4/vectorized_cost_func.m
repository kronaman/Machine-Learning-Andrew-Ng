function [J, grad] = vectorized_cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

m = length(y);
Theta1 = reshape(nn_params(1:hidden_layer_size * (1 + input_layer_size)), hidden_layer_size, (1 + input_layer_size));
Theta2 = reshape(nn_params((1 + hidden_layer_size * (1 + input_layer_size)):end), num_labels, (1 + hidden_layer_size));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_k = 1:num_labels;

%     Vectorized Forward propagation
X = [ones(m, 1) X];     % (5000, 401)
A_2 = sigmoid(X * Theta1');     % (5000, 25)
A_2 = [ones(m, 1) A_2];     % (5000, 26)
A_3 = sigmoid(A_2 * Theta2');       % (5000, 10)

%     Vectorized Backward propagation
Y = [y == y_k];     % (5000, 10)
D_3 = A_3 - Y;      % (5000, 10)
D_2 = (D_3 * Theta2) .* A_2 .* (1 - A_2);      % (5000, 26)
D_2 = D_2(:, 2:end);        % (5000, 25)

% cost and gradient without regularization
Delta2 = (D_3' * A_2);
Delta1 = (D_2' * X);
Theta2_grad = (1 / m) * Delta2;        % (10, 26)
Theta1_grad = (1 / m) * Delta1;        % (25, 401)
J = (-1/m) * sum((Y .* log(A_3) + (1 - Y) .* log(1 - A_3)), 'all');

%   Cost and gradient with regularization
if lambda ~= 0
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * (Theta1(:, 2:end));
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * (Theta2(:, 2:end));
    J = J + (lambda / (2 * m)) * (sumsqr(Theta1(:, 2:end)) + sumsqr(Theta2(:, 2:end)));
end

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end