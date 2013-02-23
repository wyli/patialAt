function [] = findreference(flagXorLoc)
    % 1 = using features, 0 using locations
addpath(genpath('~/documents/opt_learning/randomfeatures/'));

lskeys = dir('high/*.mat');
for i = 1:length(lskeys)
    for j = [0, 5, 20, 25, 40, 50, 80, 100]
        weak = sprintf(['./low/', lskeys(i).name(1:3) '_%d.mat'], j);
        keyfile = ['./high/' lskeys(i).name(1:3) '.mat'];
        rewriteReference(weak, keyfile, flagXorLoc);
    end
end


function [] = rewriteReference(weakfile, keyfile, flagXorLoc)
% load key files and weakfiles, rewrite weakfiles with nearest reference.
% usage findreference('073', '073_0');

fprintf('doing on %s\n', weakfile);
load(keyfile);
X_features = scaleFeatures(X_features')';
save(keyfile, 'X_features', 'info', 'locations');
key_loc = reshape(cell2mat(locations), 3, [])';
key_X = X_features';

load(weakfile);
X_features = scaleFeatures(X_features')';
weak_loc = reshape(cell2mat(locations), 3, [])';
weak_X = X_features';

if flagXorLoc == 1
    [~, nearestInd] = min(dist2(weak_X, key_X), [], 2);
else
    [~, nearestInd] = min(dist2(weak_loc, key_loc), [], 2);
end

locdist = zeros(size(weak_loc, 1), 1);
distanceweak = zeros(size(weak_loc, 1), 1);
for i = 1:size(weak_loc, 1)
    locdist(i) = dist2(weak_loc(i, :), key_loc(nearestInd(i), :));
    distanceweak(i) = dist2(weak_X(i, :), key_X(nearestInd(i), :));
end
save(weakfile, 'X_features', 'info', 'locations', 'nearestInd', 'locdist', 'distanceweak');
