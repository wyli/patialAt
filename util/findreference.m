function [] = findreference
addpath(genpath('~/documents/opt_learning/randomfeatures/'));

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
% usage findreference('073', '073_0');

fprintf('doing on %s\n', weakfile);
load(keyfile);
key = reshape(cell2mat(locations), 3, [])';
load(weakfile);
weakloc = reshape(cell2mat(locations), 3, [])';
[~, nearestInd] = min(dist2(weakloc, key), [], 2);
save(weakfile, 'X_features', 'info', 'locations', 'nearestInd');
