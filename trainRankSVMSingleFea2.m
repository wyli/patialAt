function [acc, nr_points] = trainRankSVMSingleFea2(schemeInd)
%matlabpool 4;
RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', sum(100*clock)));
addpath(genpath('~/documents/opt_learning/randomfeatures'));
addpath(genpath('~/dropbox/libr/matlab'));
load('../../exparam.mat');
which train
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

indexes = [];
for i = 1:50
    indexes = [indexes, i:50:2450];
end

for i = 53:60:1800

    cInd = indexes(1:i);
    [fea, y] = calculateTrainingSet(...
        feaHigh, trainYHigh, highImageInd, referHighInd,...
        feaLow, trainYLow, lowImageInd, referLowInd, cInd);
    [accs, aucs, score_column] = expRankSVM(...
        fea, y,...
        feaTest, testY);
    auc = [auc, aucs];
    acc = [acc, accs];
    scores = [scores, score_column];
    nr_points = [nr_points, i];
    fprintf('!!!i = %d, acc = %f, auc = %f', i, accs, aucs);
    isave(['huberscoressinglefea', num2str(i)], score_column, i);
end
save('huberrankSVMsinglefea', 'acc', 'auc', 'nr_points', 'scores');
end %end of trainweakSVMfunction

function isave(name, x, i)
save(name, 'x', 'i');
end

function [acc, auc, scores] = expRankSVM(...
        featureSet, y,...
        feaTest, testY)

fprintf('size of training: %d\n', size(featureSet, 1));
facty = abs(y(1));
accnow = 0;
bestcmd = [];
for log10c = [-5]
    for log10p = [-9]
        cmd = ['-s 2 -c ', num2str(10^log10c), ' -p ', num2str(10^log10p)];
        modelnow = train(sparse(y), sparse(featureSet), cmd);
        [~, ~, scores] = predict(sparse(y), sparse(featureSet), modelnow, '-q');
        try
            [~, ~, ~, auc] = perfcurve((y>0)*2 -1, scores, 1);
        catch
            scores
            auc = 0;
        end
        if (auc > accnow)
            bestcmd = cmd;
            accnow = auc;
        end
    end
end

fprintf('bestcmd1: %s\n', bestcmd);
%modelbest = train(sparse(y), sparse(featureSet), bestcmd);
clear y featureSet cmd bestcmd

testY = double(testY > 0) * 2 - 1;
[~, acc, scores] = predict(sparse(testY), sparse(feaTest), modelnow);
scores(isnan(scores)) = 0;
try
    [~, ~, ~, auc] = perfcurve(testY, scores, '1')
catch
    fprintf('????????????why???\n');
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
    feadists = exp(feadists/std(feadists));
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
