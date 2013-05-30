function [acc,nr_points]  = trainConventionalSVM(schemeInd)
%RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', sum(100*clock)));
addpath(genpath('~/documents/opt_learning/randomfeatures'));
addpath(genpath('~/desktop/liblinear-1.93/matlab'));
load('../../exparam.mat');
which train

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
scores = [];
total = min(sum(trainYHigh > 0), sum(trainYHigh < 0));
indexes = [];
for i = 1:50
    indexes = [indexes, i:50:2450];
end

for i = length(indexes)

    cInd = indexes(1:i);
    [accs, aucs, score_column]= expConventionalSVM(...
        feaHigh(cInd, :),...
        trainYHigh(cInd),...
        feaTest, testY);
    auc = [auc, aucs];
    acc = [acc, accs];
    nr_points = [nr_points, i];
    scores = [scores, score_column];
end
save('consvm', 'acc', 'auc', 'nr_points', 'scores');
end %end of function



function [acc, auc, scores] = expConventionalSVM(...
        featureSet, y,...
        feaTest, testY)

accnow = 0;
bestcmd = [];
for log10c = -5:-1:-10
    for log10e = 5:-1:-5
    cmd = ['-s 2 -c ', num2str(10^log10c), ' -e ' num2str(10^log10e)];
    modelnow = train(sparse(y), sparse(featureSet), [cmd ' -q']);
    [~, ~, scores] = predict(sparse(y), sparse(featureSet), modelnow, [' -q']);
    scores(isnan(scores)) = 0;
    [~, ~, ~, auc] = perfcurve(y, scores, '1');
    if (auc > accnow && sum(modelnow.w) ~= 0)
        bestcmd = cmd;
        accnow = auc;
    end
end
end
fprintf('training auc: %f cmd: %s\n', auc, bestcmd);
modelbest = train(sparse(y), sparse(featureSet), [bestcmd ' -q']);
clear y log10e tempmodel scores auc aucnow cmd featureSet bestcmd

[~, acc, scores] = predict(sparse(testY), sparse(feaTest), modelbest);
try
    [~, ~, ~, auc] = perfcurve(testY, scores, 1)
catch
    auc = 0;
end
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
