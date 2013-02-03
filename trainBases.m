function [] = trainBases(...
        baseDir, cuboidSet, trainInd, windowSize, subSize, step3d, k, randMat)
fprintf('%s find %d clusters on small window %d\n', datestr(now), k, subSize);
% params
global numOfSubsamples
numOfSubsamples = 6;
samplesPerFile = 800;
% input
cuboidSet = [cuboidSet '/cuboid_%d/high/%s'];

% output
clusterFile = [baseDir, '/clusters.mat'];

localSet = [];
listFiles = dir(sprintf(cuboidSet, windowSize, '*.mat'));
for j = 1:size(trainInd, 1)
    i = trainInd(j);
    cuboidFile = sprintf(cuboidSet, windowSize, listFiles(i).name);
    load(cuboidFile);
    r = randsample(size(cuboid,2), min(size(cuboid,2), samplesPerFile));
    cuboid = cuboid(1,r);

    idMat = ones(1, size(cuboid, 2));
    repSize = mat2cell(idMat.*subSize, 1, idMat);
    repStep = mat2cell(idMat.*step3d, 1, idMat);
    localCells = cellfun(@sampleSubCuboids,...
        cuboid, repSize, repStep, 'UniformOutput', false);
    localMat = cell2mat(localCells');
    clear localCells cuboid;
    localSet = [localSet; localMat];
end
localSet = (randMat*localSet')';
assert(size(localSet, 1) > 40000, '%d %d', size(localSet, 1), size(localSet, 2));
r = randsample(size(localSet, 1), 40000);
localSet = localSet(r, :);
prm.nTrial = 3;
prm.maxIter = 200;
[~, clusters] = kmeans2(localSet, k, prm);
save(clusterFile, 'clusters');
end

function localCuboid = sampleSubCuboids(image3d, wSize, wStep)
global numOfSubsamples;
imgSize = size(image3d);
halfSize = ceil(wSize/2);

xs = halfSize:wStep:(imgSize(1) - halfSize);
ys = halfSize:wStep:(imgSize(2) - halfSize);
zs = halfSize:wStep:(imgSize(3) - halfSize);

xrec = min(length(xs), numOfSubsamples);
yrec = min(length(ys), numOfSubsamples);
zrec = min(length(zs), numOfSubsamples);

xs = randsample(xs, xrec);
ys = randsample(ys, yrec);
zs = randsample(zs, zrec);

localCuboid = zeros(numel(xs), wSize^3);
for i = 1:numel(xs)
    sampleCell = getSurroundCuboid(...
        image3d, [xs(i), ys(i), zs(i)], [wSize, wSize, wSize]);
    localCuboid(i, :) = sampleCell(:)';
end
end
