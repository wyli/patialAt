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
scores = [];
total = min(sum(trainYHigh > 0), sum(trainYHigh < 0));
for i = 2:10:total

    [accs, aucs, score_column]= expConventionalSVM(...
        [feaHigh(1:i, :); feaHigh(end-i+1:end, :)],...
        [trainYHigh(1:i); trainYHigh(end-i+1:end)],...
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
for log10c = 7:-1:-7
    cmd = ['-s 2 -c ', num2str(10^log10c)];
    acc = train(sparse(y), sparse(featureSet), [cmd ' -v 3 -q']);
    if (acc >= accnow)
        bestcmd = cmd;
        accnow = acc;
    end
end
fprintf('training auc: %f cmd: %s\n', acc, bestcmd);
modelbest = train(sparse(y), sparse(featureSet), bestcmd);
clear y log10e tempmodel scores auc aucnow cmd featureSet bestcmd

[~, acc, scores] = predict(sparse(testY), sparse(feaTest), modelbest);
try
    [~, ~, ~, auc] = perfcurve(testY, scores, 1);
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