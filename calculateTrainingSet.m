function [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, cInd)

    fprintf('filtered features, low size %d\n', length(trainYLow));
    fea = feaHigh(cInd, :);
    y = trainYHigh(cInd);
    indAll = zeros(size(trainYLow));
    for j = cInd
        indAll = indAll +...
            double((lowImageInd == highImageInd(j)) & (referLowInd == referHighInd(j)));
    end
    fea = [fea; feaLow(indAll > 0, :)];
    y = [y; trainYLow(indAll > 0)];
end
