function [] = findreference(resultInd, click)
    % scale features, add references and distances

%addpath(genpath('~/documents/opt_learning/randomfeatures/'));
dirString = '~/output/exp_20130529T185037_primal/result_%02d/fea%d';
currentDir = sprintf(dirString, resultInd, click);
cd(currentDir);
pwd

lskeys = dir('high/*.mat');
for i = 1:length(lskeys)
    for j = [0, 5, 20, 25, 40, 50, 80, 100]
        weak = sprintf(['./low/', lskeys(i).name(1:3) '_%d.mat'], j);
        keyfile = ['./high/' lskeys(i).name(1:3) '.mat'];
        rewriteReference(weak, keyfile);
    end
end


function [] = rewriteReference(weakfile, keyfile)
% load key files and weakfiles, rewrite weakfiles with nearest reference.
% usage findreference('low/073_0', 'high/073');

fprintf('doing on %s\n', weakfile);
load(keyfile);
X_features = single(X_features);
X_features = scaleFeatures(X_features')';
save(keyfile, 'X_features', 'info', 'locations');
key_loc = reshape(cell2mat(locations), 3, [])';
key_X = X_features';

load(weakfile);
X_features = single(X_features);
X_features = scaleFeatures(X_features')';
weak_loc = reshape(cell2mat(locations), 3, [])';
weak_X = X_features';

[~, feaRef] = min(dist2(weak_X, key_X), [], 2);
[~, locRef] = min(dist2(weak_loc, key_loc), [], 2);

feaDist = zeros(size(weak_loc, 1), 1);
locDist = zeros(size(weak_loc, 1), 1);
for i = 1:size(weak_loc, 1)
    feaDist(i) = dist2(weak_X(i, :), key_X(feaRef(i), :));
    locDist(i) = dist2(weak_loc(i, :), key_loc(locRef(i), :));
end
save(weakfile, 'X_features', 'info', 'locations', 'feaRef', 'feaDist', 'locRef', 'locDist');
