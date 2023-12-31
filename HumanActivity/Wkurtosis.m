function K = Wkurtosis(X)
%AVERAGEABSOLUTEDIFFERENCE Average absolute difference between the value
% of each of the 200 readings within the ED and the mean value over those 200 values 
%(for each axis) 

    K = kurtosis(X, 0, 2);

    %(sum((x-mean(x)).^4)./length(x)) ./ (var(x,1).^2)
    
end

