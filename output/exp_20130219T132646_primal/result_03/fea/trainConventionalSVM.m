function [acc,nr_points]  = trainConventionalSVM(schemeInd)

addpath(genpath('~/documents/opt_learning/randomfeatures'));
addpath(genpath('~/desktop/liblinear-1.93/matlab'));
load('../../exparam.mat');

trainInd = allInd(:, ~testScheme(schemeInd, :));
trainInd = trainInd(:);
trainInd = trainInd(trainInd < 60);

testInd = allInd(:, testScheme(schemeInd, :) == 1);
testInd = testInd(:);
testInd = testInd(testInd < 60);

% load training set
[feaHigh, trainYHigh] = loadFeatures('./high', trainInd);
% load testing set
[feaTest, testY] = loadFeatures('../testingfea', testInd);

auc = [];
acc = [];
nr_points = [];
total = min(sum(trainYHigh > 0), sum(trainYHigh < 0));
for i = 2:1:120

    [accs, aucs]= expConventionalSVM(...
        [feaHigh(1:i, :); feaHigh(end-i+1:end, :)],...
        [trainYHigh(1:i); trainYHigh(end-i+1:end)],...
        feaTest, testY);
    auc = [auc, aucs];
    acc = [acc, accs];
    nr_points = [nr_points, i];
end
save('consvm', 'acc', 'auc', 'nr_points');
end %end of function



function [acc, auc] = expConventionalSVM(...
        featureSet, y,...
        feaTest, testY)
[scaledFeatures, scaleVectors] = scaleFeatures(featureSet, [], -1);
clear featureSet;

accnow = 0;
bestcmd = [];
for log10p = -1:-1:-7
    for log10e = -1:-1:-7
        cmd = ['-s 2 -c 0', ' -e ', num2str(10^log10e), ' -p ',  num2str(10^log10p), ' -q'];
        tempmodel = train1(sparse(y), sparse(scaledFeatures), cmd);
        [~, ~, scores] = predict1(sparse(y), sparse(scaledFeatures), tempmodel, '-q');
        scores(isnan(scores)) = 0;
        [~, ~, ~, acc] = perfcurve(y, scores, '1');
        if ((acc > accnow) && (sum(tempmodel.w) ~= 0))
            bestcmd = cmd;
            accnow = acc;
        end
    end
end
fprintf('training auc: %f cmd: %s\n', acc, bestcmd);
modelbest = train1(sparse(y), sparse(scaledFeatures), bestcmd);
clear y log10e tempmodel scores auc aucnow cmd scaledFeatures bestcmd

[scaledFeatures, ~] = scaleFeatures(feaTest, scaleVectors, 1);
clear feaTest scaleVectors;
[~, acc, scores] = predict1(sparse(testY), sparse(scaledFeatures), modelbest);
[~, ~, ~, auc] = perfcurve(testY, scores, '1');
end % end of expConventionalSVM



function [features, y] = loadFeatures(setpath, ind)

lsFiles = dir([setpath '/*.mat']);
feaHigh = [];
yHigh = [];
for i = 1:length(ind)

    fprintf([setpath '/%s\n'], lsFiles(ind(i)).name);
    load([setpath '/', lsFiles(ind(i)).name]);
    feaHigh = [feaHigh, X_features];
    yHigh = [yHigh, info(1, :)];
end
clear X_features lsFiles locations info;

features = feaHigh';

y = zeros(size(yHigh))';
for i = 1:length(yHigh)

    if strcmp(yHigh{i}.type, 'LGD')
        y(i) = -1;
    else
        y(i) = 1;
    end
end
end
