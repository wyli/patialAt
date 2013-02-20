function [modelNew, scaleVectors, trainAcc] =...
        trainClassifier(feaPathHigh, feaPathLow, trainInd, radius)

% load high confidence features
lsFiles = dir([feaPathHigh '/*.mat']);
feaHigh = [];
yHigh = [];
for i = 1:length(trainInd)
    %load([feaPathHigh '/' lsFiles(trainInd(1)).name]);
    fprintf([feaPathHigh '/' lsFiles(trainInd(1)).name]);
    feaHigh = [feaHigh, X_features];
    yHigh = [yHigh, info{ind}];
end

lowRadiusPath = sprintf(feaPathHigh, '/', '*_', radius, '.mat');
lsFiles = dir(lowRadiusPath);
feaLow = [];
yLow = [];
for i = 1:length(trainInd)
    %load([feaPathLow '/' lsFiles(trainInd(1)).name, '_', radius, '.mat']);
    fprintf([feaPathLow '/' lsFiles(trainInd(1)).name, '_', radius, '.mat']);
    feaLow = [feaLow, X_features];
    yLow = [yLow, info{ind}];
end

feaHigh = feaHigh';
feaLow = feaLow';
 

trainYHigh = zeros(size(yHigh))';
for i = 1:length(yHigh)
    if strcmp(yHigh(i).type,'LGD')
        trainYHigh(i) = -1;
    else
        trainYHigh(i) = 1;
    end
end
clear yHigh;

trainYLow = zeros(size(yLow))';
for i = 1:length(yLow)
    if strcmp(yLow(i).type,'LGD')
        trainYLow(i) = -0.75;
    else
        trainYLow(i) = 0.75;
    end
end
clear yLow;

[scaledFeatures, scaleVectors] = scaleFeatures([feaHigh; feaLow], [], -1);
modelNew = train(sparse([trainYHigh; trainYLow]),...
    sparse(scaledFeatures), sprintf('-s %d -e %.10f', type, cost));
[trainAcc, ~] = predict(sparse(0), sparse(scaledFeatures), modelNew, '-q');

%for i = 0.1
%trainall = [trainHigh; trainLow];
%c = [ones(size(trainHigh, 1), 1) *0.0001;  ones(size(trainLow, 1), 1) * .1];
%model = train(sparse([trainYHigh; trainYLow]), sparse([trainall, c]), '-s 2');
% trainall = trainHigh;
% c = ones(size(trainHigh, 1),1);
% model = train(sparse(trainYHigh), sparse([trainall, c]), '-s 2');

%[label, acc, dec]= predict(sparse(testYHigh), sparse(testHigh), model);
% acc1 = [acc1, acc];

% model = train(sparse([trainYHigh]), sparse([trainHigh]), '-e 0.00001 -s 2');
% [label, acc, dec]= predict(sparse([testYHigh]), sparse([testHigh]), model);
%end
end
