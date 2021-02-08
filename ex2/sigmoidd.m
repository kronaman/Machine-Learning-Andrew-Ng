function g = sigmoidd(z)
    %UNTITLED Summary of this function goes here
    %   Detailed explanation goes here
    g = zeros(size(z));
    if isscalar(z) || isvector(z)
        g = 1 ./ (1 + exp(-z));
    else   % matrix
        for col = 1:size(z,2)
            g(:,col) = 1 ./ (1 + exp(-z(:,col)));
        end
    end
end