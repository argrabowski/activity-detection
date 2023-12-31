function E = Wentropy(X)
%ENTROPY Summary of this function goes here
    
    E = [];
    
    for i = 1:size(X,1)
        xRow = X(i,:);
        p = histcounts(xRow) / sum(histcounts(xRow));
        E = [E; -sum(p.* log2(p))];
    end
end

