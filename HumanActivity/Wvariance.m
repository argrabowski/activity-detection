function Y = Wvariance(X)
    A = mean(X,2);
    S = size(X, 2);
    ySum = sum((X - A).^2, 2);
    Y = ((1/(S-1)) * ySum);
end