function g = sigmoid(z)
    %SIGMOID Compute sigmoid function
    %   g = SIGMOID(z) computes the sigmoid of z.
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    %               vector or scalar).
    
    g = zeros(size(z));
    if isscalar(z) || isvector(z)
        g = 1 ./ (1 + exp(-z));
    else   % matrix
        for col = 1:size(z,2)
            g(:,col) = 1 ./ (1 + exp(-z(:,col)));
        end
    end
    
    % =============================================================
end
