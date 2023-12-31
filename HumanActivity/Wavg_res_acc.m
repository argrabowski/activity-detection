function Y = Wavg_res_acc(X, Y, Z)
    % Calculate the square of the values of each axis
    squared_X = X.^2;
    squared_Y = Y.^2;
    squared_Z = Z.^2;
    
    % Calculate the sum of the squares for each row
    squared_sum = sum(squared_X + squared_Y + squared_Z, 2);
    
    % Calculate the square root of the sum of squares
    sqrt_sum_squares = sqrt(squared_sum);
    
    % Calculate the mean of the square roots
    Y = sqrt_sum_squares / size(X(1,:), 2);
end