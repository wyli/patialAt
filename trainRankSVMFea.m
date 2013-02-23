function [acc, nr_points] = trainRankSVM(schemeInd)
matlabpool 4;
RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', sum(100*clock)));
addpath(genpath('~/documents/opt_learning/randomfeatures'));
addpath(genpath('~/dropbox/libr/matlab'));
load('../../exparam.mat');

trainInd = allInd(:, ~testScheme(schemeInd, :));
trainInd = trainInd(:);
trainInd = trainInd(trainInd < 60);

testInd = allInd(:, testScheme(schemeInd, :) == 1);
testInd = testInd(:);
testInd = testInd(testInd < 60);

[feaHigh, trainYHigh, highImageInd, referHighInd] =...
    loadFeaturesWithRadius('./high', trainInd, -1);
feaLow = [];
trainYLow = [];
lowImageInd = [];
referLowInd = [];
for r = [0, 5, 20, 25, 40, 50, 80, 100]
    [tempfeaLow, temptrainYLow, templowImageInd, tempreferLowInd] =...
        loadFeaturesWithRadius('./low', trainInd, r);
    feaLow = [feaLow; tempfeaLow];
    trainYLow = [trainYLow; temptrainYLow];
    lowImageInd = [lowImageInd; templowImageInd];
    referLowInd = [referLowInd; tempreferLowInd];
end
[feaTest, testY, ~, ~] =...
    loadFeaturesWithRadius('../testingfea', testInd, -1);

auc = [];
acc = [];
scores = [];
nr_points = [];
total = min(sum(trainYHigh > 0), sum(trainYHigh < 0));
parfor j = 1:length(2:10:total)%2:1:120
    indexes=2:10:total;
    i = indexes(j);
    [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, i);
    [accs, aucs, score_column] = expRankSVM(...
        fea, y,...
        feaTest, testY);
    auc = [auc, aucs];
    acc = [acc, accs];
    scores = [scores, score_column];
    nr_points = [nr_points, i];
end
save('rankSVM', 'acc', 'auc', 'nr_points', 'scores');
end %end of trainweakSVMfunction

function [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, i)

    fprintf('filtered features\n');
    fea = [feaHigh(1:i, :); feaHigh(end-i+1:end, :)];
    y = [trainYHigh(1:i, :); trainYHigh(end-i+1:end, :)];
    indAll = zeros(size(trainYLow));
    for j = [1:i, length(trainYHigh)-i+1:length(trainYHigh)]
        indAll = indAll +...
            double((lowImageInd == highImageInd(j)) & (referLowInd == referHighInd(j)));
    end
    fea = [fea; feaLow(indAll > 0, :)];
    y = [y; trainYLow(indAll > 0)];
    
end % end of calculateTrainingSet


function [acc, auc, scores] = expRankSVM(...
        featureSet, y,...
        feaTest, testY)

fprintf('size of training: %d\n', size(featureSet, 1));
facty = abs(y(1));
accnow = 0;
bestcmd = [];
for log10c = [-16, -12]
    for log10p = [-18, -16, -14]
        cmd = ['-s 0 -c ', num2str(10^log10c), ' -p ', num2str(10^log10p)];
        yy = y;
        yy(y<0) = -facty;
        acc = train1(sparse(yy), sparse(featureSet), [cmd ' -v 3 -q']);
        if (acc > accnow)
            bestcmd = cmd;
            accnow = acc;
        end
    end
end
accnow = 0;
bestcmd2 = [];
for log10c = [-16, -12]
    for log10p = [-18, -16, -14]
        cmd = ['-s 0 -c ', num2str(10^log10c), ' -p ', num2str(10^log10p)];
        yy = y;
        yy(y>0) = facty;
        acc = train1(sparse(yy), sparse(featureSet), [cmd ' -v 3 -q']);
        if (acc > accnow)
            bestcmd2 = cmd;
            accnow = acc;
        end
    end
end

fprintf('bestcmd1: %s\n', bestcmd);
fprintf('bestcmd2: %s\n', bestcmd2);
tempy = y;
tempy(y<0) = -1;
modelpos = train1(sparse(tempy), sparse(featureSet), bestcmd);
tempy = y;
tempy(y>0) = 1;
modelneg = train1(sparse(tempy), sparse(featureSet), bestcmd2);
clear y featureSet cmd bestcmd

testY = (testY > 0);
[~, ~, scoresneg] = predict1(sparse(testY), sparse(feaTest), modelneg);
[~, ~, scorespos] = predict1(sparse(testY), sparse(feaTest), modelpos);
[scores, labels] = max([scoresneg, scorespos], [], 2);
labels = labels - 1;
prelabels = (labels == testY);
acc = 100 * sum(prelabels) / length(prelabels);
try
    [~, ~, ~, auc] = perfcurve(testY, scores, '1')
catch
    auc = 0;
end
end %end of expranksvm

function [features, y, imageInd, referInd] =...
        loadFeaturesWithRadius(setpath, ind, radius)
info = [];
if radius > -1
    filesetname = sprintf([setpath '/*_%d.mat'], radius);
else
    filesetname = sprintf([setpath '/*.mat']);
end

lsFiles = dir(filesetname);
feaHigh = [];
yHigh = [];
imageInd = [];
referInd = [];
feadists = [];
for i = 1:length(ind)

    if radius > -1
        fprintf([setpath '/%s\n'], lsFiles(ind(i)).name)
        load([setpath '/', lsFiles(ind(i)).name]);
        r = randsample(size(X_features, 2), 20);
        feaHigh = [feaHigh, X_features(:, r)];
        yHigh = [yHigh, info(1, r)];
        imageInd = [imageInd; ones(size(info(1,r)))' * i];
        referInd = [referInd; nearestInd(r)];
        feadists = [feadists; distanceweak(r)];
    else
        fprintf([setpath '/%s\n'], lsFiles(ind(i)).name)
        load([setpath '/', lsFiles(ind(i)).name]);
        feaHigh = [feaHigh, X_features];
        yHigh = [yHigh, info(1, :)];
        imageInd = [imageInd; ones(size(info(1,:)))' * i];
        referInd = [referInd; (1:length(info(1, :)))'];
    end
end
clear X_features lsFiles locations info;

features = feaHigh';

y = zeros(size(yHigh))';
if radius > -1
    feadists = exp(feadists/std(feadists)) - 0.5;
    for i = 1:length(yHigh)

        if strcmp(yHigh{i}.type, 'LGD')
            y(i) = -feadists(i);
        else
            y(i) = feadists(i);
        end
    end
else
    for i = 1:length(yHigh)

        if strcmp(yHigh{i}.type, 'LGD')
            y(i) = -0.5;
        else
            y(i) = 0.5;
        end
    end
end
end % end of loadFeaturesWithRadius
