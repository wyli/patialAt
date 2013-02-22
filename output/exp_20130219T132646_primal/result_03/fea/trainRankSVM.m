function [acc, nr_points] = trainRankSVM(schemeInd)
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
nr_points = [];
total = min(sum(trainYHigh > 0), sum(trainYHigh < 0));
for i = 2:1:120

    [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, i);
    [accs, aucs] = expRankSVM(...
        fea, y,...
        feaTest, testY);
    auc = [auc, aucs];
    acc = [acc, accs];
    nr_points = [nr_points, i];
end
save('rankSVM', 'acc', 'auc', 'nr_points');
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
end


function [acc, auc] = expRankSVM(...
        featureSet, y,...
        feaTest, testY)
[scaledFeatures, scaleVectors] = scaleFeatures(featureSet, [], -1);
clear featureSet;

fprintf('size of training: %d\n', size(scaledFeatures, 1));
accnow = 0;
bestcmd = [];
%for log10c = 1:-1:-4
    %for log10p = [-1, -3]
        cmd = ['-s 0 -c 0.1 -e  0.0001 -p 0.1 -q'];
        yy = y;
        yy(y<0) = -1;
        tempmodel = train1(sparse(yy), sparse(scaledFeatures), cmd);
        [~, ~, scores] = predict1(sparse(yy), sparse(scaledFeatures), tempmodel, '-q');
        scores(isnan(scores)) = 0;
        binaryY = (y > 0)*2 - 1;
        [~, ~, ~, acc] = perfcurve(binaryY, scores, '1');
        if ((acc > accnow) & (sum(tempmodel.w) ~= 0))
            bestcmd = cmd;
            accnow = acc;
        end
    %end
%end
accnow = 0;
bestcmd2 = [];
%for log10c = 1:-1:-4
    %for log10p = [-1, -3]
        cmd = ['-s 0 -c 0.1 -e  0.0001 -p 0.1 -q'];
        yy = y;
        yy(y>0) = 1;
        tempmodel = train1(sparse(yy), sparse(scaledFeatures), cmd);
        [~, ~, scores] = predict1(sparse(y), sparse(scaledFeatures), tempmodel, '-q');
        scores(isnan(scores)) = 0;
        binaryY = (y > 0)*2 - 1;
        [~, ~, ~, acc] = perfcurve(binaryY, scores, '1');
        if ((acc > accnow) & (sum(tempmodel.w) ~= 0))
            bestcmd2 = cmd;
            accnow = acc;
        end
    %end
%end

fprintf('bestcmd1: %s\n', bestcmd);
fprintf('bestcmd2: %s\n', bestcmd2);
tempy = y;
tempy(y<0) = -1;
modelbest1 = train1(sparse(tempy), sparse(scaledFeatures), bestcmd);
tempy = y;
tempy(y>0) = 1;
modelbest2 = train1(sparse(tempy), sparse(scaledFeatures), bestcmd2);
clear y log10c log10e tempmodel scores auc aucnow cmd scaledFeatures bestcmd

[scaledFeatures, ~] = scaleFeatures(feaTest, scaleVectors, 1);
clear feaTest scaleVectors;
testY = (testY > 0) * 2 - 1;
[~, ~, scores1] = predict1(sparse(testY), sparse(scaledFeatures), modelbest1);
[~, ~, scores2] = predict1(sparse(testY), sparse(scaledFeatures), modelbest2);
scores = max(scores1, scores2);
prelabels = (scores1>scores2) & (testY > 0);
acc = sum(prelabels) / length(prelabels);
try
    [~, ~, ~, auc] = perfcurve(testY, scores, '1')
catch
    auc = 0;
end
end % end of expConventionalSVM



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
for i = 1:length(ind)

    if radius > -1
        fprintf([setpath '/%s\n'], lsFiles(ind(i)).name)
        load([setpath '/', lsFiles(ind(i)).name]);
        r = randsample(size(X_features, 2), 20);
        feaHigh = [feaHigh, X_features(:, r)];
        yHigh = [yHigh, info(1, r)];
        imageInd = [imageInd; ones(size(info(1,r)))' * i];
        referInd = [referInd; nearestInd(r)];
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
    for i = 1:length(yHigh)

        if strcmp(yHigh{i}.type, 'LGD')
            y(i) = -(radius/5 + 0.25);
        else
            y(i) = radius/5 + 0.25;
        end
    end
else
    for i = 1:length(yHigh)

        if strcmp(yHigh{i}.type, 'LGD')
            y(i) = -0.05;
        else
            y(i) = 0.05;
        end
    end
end
end
