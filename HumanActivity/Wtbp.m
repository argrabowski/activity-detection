function Y = Wtbp(X)
    timeBetweePoints = (2.5 / 128) * 1000; %One row is 128 entries captured in 2.5 seconds SO 19.5ms between each entry 
    Y = zeros(size(X,1), 1);
    for i = 1: size(X,1)
       xRow = X(i, :);
       [peakValues, peakLocations] = findpeaks(xRow); 
       msTimes = peakLocations*timeBetweePoints;
       timeDiff = diff(msTimes);
       Y(i) = mean(timeDiff);
    end
end

