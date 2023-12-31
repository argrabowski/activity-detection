function Y = Waxes_corr(X_data, Y_data)
    s = size(X_data,1);
    corrArr = [];
    for i = 1:s
        xRow = X_data(i,:);
        yRow = Y_data(i,:);
        corrVal = corrcoef(xRow, yRow);
        corrArr = [corrArr;corrVal(2:2)];
    end
    Y = corrArr;
end