function Y = Wmean_abs_dev(X)
    % Calculate the mean of X
    mean_X = mean(X, 2);

    % Calculate the absolute deviations from the mean
    abs_dev = abs(X - mean_X);

    % Calculate the sum of absolute deviations
    dev_sum = sum(abs_dev, 2);

    % Calculate the MAD
    Y = sqrt(dev_sum / (length(X(1,:)) - 1));
end