function Y = Wbinned_dist(X)
    s = size(X,1);
    countArr = [];
    for i = 1:s
        xRow = X(i,:);
        m_min = min(xRow);
        m_max = max(xRow);
        space = flip(linspace(m_max, m_min, 11), 2);
        space(:,1) = [];
        count = [0,0,0,0,0,0,0,0,0,0];
        s2 = size(xRow,2);
        for k = 1:s2
            current = xRow(1,k);
            for j = 1:size(space, 2)
                spaceVal = space(1,j);
                if current <= spaceVal
                    count(1,j) = count(1,j) + 1;
                    break
                end
            end
        end
        countArr = [countArr; count];
        histcounts(xRow, 10);
    end
    Y = countArr;
end