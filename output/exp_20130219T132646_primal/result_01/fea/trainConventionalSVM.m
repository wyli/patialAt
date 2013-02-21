function [] = trainConventionalSVM(schemeInd)

addpath(genpath('~/documents/opt_learning/randomfeatures'));
addpath(genpath('~/desktop/liblinear-1.93/matlab'));
load('../../exparam.mat');
lsFiles = dir('./high/*.mat');

trainInd = allInd(:, ~testScheme(schemeInd, :));
trainInd = trainInd(:);
trainInd = trainInd(trainInd < 60);

feaHigh = [];
yHigh = [];
for i = 1:length(trainInd)

    fprintf('./high/%s\n', lsFiles(trainInd(i)).name);
    load(['./high/', lsFiles(trainInd(i)).name]);
    feaHigh = [feaHigh, X_features];
    yHigh = [yHigh, info{i}];
end
clear X_features lsFiles trainInd allInd testScheme schemeInd locations info;

feaHigh = feaHigh';

trainYHigh = zeros(size(yHigh))';
for i = 1:length(yHigh)

    if strcmp(yHigh(i).type, 'LGD')
        trainYHigh(i) = -1;
    else
        trainYHigh(i) = 1;
    end
end
clear yHigh i;


[scaledFeatures, scaleVectors] = scaleFeatures(feaHigh, [], -1);
end %end of function
