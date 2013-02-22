function [auc, nr_points] = trainWeakSVM(schemeInd)
RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', sum(100*clock)));
addpath(genpath('~/documents/opt_learning/randomfeatures'));
%addpath(genpath('~/desktop/liblinear-1.93/matlab'));
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

acc = [];
auc = [];
nr_points = [];
total = min(sum(trainYHigh > 0), sum(trainYHigh < 0));
for i = 2:1:120

    [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, i);
    [accs, aucs] = expConventionalSVM(...
        fea, y,...
        feaTest, testY);
    auc = [auc, aucs];
    acc = [acc, accs];
    nr_points = [nr_points, i];
end
save('weaksvm', 'acc', 'auc', 'nr_points');
end %end of trainweakSVMfunction

function [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, i)

    fprintf('filtered features, low size %d\n', length(trainYLow));
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


function [acc, auc] = expConventionalSVM(...
        featureSet, y,...
        feaTest, testY)
[scaledFeatures, scaleVectors] = scaleFeatures(featureSet, [], -1);
clear featureSet;

fprintf('size of training: %d\n', size(scaledFeatures, 1));
accnow = 0;
bestcmd = [];
for log10e = -1:-1:-7
    for log10p = [-1, -3]
        cmd = ['-s 2 -c 0 -e ', num2str(10^log10e), ' -p ', num2str(10^log10p), ' -q'];
        tempmodel = train1(sparse(y), sparse(scaledFeatures), cmd);
        [~, ~, scores] = predict1(sparse(y), sparse(scaledFeatures), tempmodel, '-q');
        scores(isnan(scores)) = 0;
        [~, ~, ~, acc] = perfcurve(y, scores, '1');
        if ((acc > accnow) & (sum(tempmodel.w) ~= 0))
            bestcmd = cmd;
            accnow = acc;
        end
    end
end
fprintf('training acc: %f cmd: %s\n', acc(1), bestcmd);
modelbest = train1(sparse(y), sparse(scaledFeatures), bestcmd);
clear y log10e tempmodel scores aucnow cmd scaledFeatures bestcmd

[scaledFeatures, ~] = scaleFeatures(feaTest, scaleVectors, 1);
clear feaTest scaleVectors;
[~, acc, scores] = predict1(sparse(testY), sparse(scaledFeatures), modelbest);
try
    [~, ~, ~, auc] = perfcurve(testY, scores, '1');
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
for i = 1:length(yHigh)

    if strcmp(yHigh{i}.type, 'LGD')
        y(i) = -1;
    else
        y(i) = 1;
    end
end
end
