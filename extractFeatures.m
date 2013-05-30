function [] = extractFeatures(...
        baseSet, cuboidSet, feaSet, ...
        windowSize, subSize, step3d, randMat, clicks)
global projMat
projMat = randMat;
fprintf('%s build histogram for each cuboid\n', datestr(now));
% input
clusterSet = [baseSet, '/clusters_' int2str(clicks)];
% output
mkdir(feaSet);
feaSet = [feaSet '/%s'];

tempclusters = load(clusterSet);
clusters = tempclusters.clusters;
listFiles = dir(sprintf(cuboidSet, windowSize, '*.mat'));
for i = 1:size(listFiles, 1)
    fprintf('%s extracting BoG features %s\n', datestr(now), listFiles(i).name);
    cuboidFile = sprintf(cuboidSet, windowSize, listFiles(i).name);
    tempcuboid = load(cuboidFile);
    cuboid = tempcuboid.cuboid;
%     if(samplePerFile < size(cuboid, 2))
%         r = randsample(size(cuboid, 2), min(size(cuboid, 2), samplePerFile));
%         cuboid = cuboid(:, r);
%         save(cuboidFile, 'cuboid');
%     end
    locations = cuboid(2,:);
    info = cuboid(3,:);
%    cuboid = cuboid(1,:);


    %idMat = ones(1, size(cuboid, 2));
    %repSize = mat2cell(idMat.*subSize, 1, idMat);
    %repStep = mat2cell(idMat.*step3d, 1, idMat);
%    rMat = ones(1, size(cuboid, 2)) * size(clusters, 2);
%    repClusters = mat2cell(...
%        repmat(clusters, 1, size(cuboid,2)), size(clusters, 1), rMat);

%     histograms = cellfun(@cuboid2Hist,...
%         cuboid, repClusters, repSize, repStep, 'UniformOutput', false);
%    histograms = cellfun(@cuboid2Hist,...
%        cuboid, repClusters, 'UniformOutput', false);
%     clear rMat repClusters repSize repStep cuboid;

    histograms = cell(1, size(cuboid,2));
    for index = 1:length(histograms)
        histograms{index} = cuboid2Hist(cuboid{4, index}, clusters);
    end
    %clear rMat repClusters cuboid;
    X_features = cell2mat(histograms');
    X_features = int16(X_features');
    featureFile = sprintf(feaSet, listFiles(i).name);
    save(featureFile, 'X_features', 'info', 'locations');
    clear X_features histograms info locations;
end
end

%function histogram = cuboid2Hist(image3d, clusters, wSize, wStep)
%function histogram = cuboid2Hist(image3d, clusters)
function histogram = cuboid2Hist(localCuboid, clusters)
%global projMat
%imgSize = size(image3d);
%halfSize = ceil(wSize/2);
%xs = halfSize:wStep:(imgSize(1) - halfSize);
%ys = halfSize:wStep:(imgSize(2) - halfSize);
%zs = halfSize:wStep:(imgSize(3) - halfSize);
%[x y z] = meshgrid(xs, ys, zs);
% [x, y, z] = meshgrid(5:2:17);
% x = x(:);
% y = y(:);
% z = z(:);
% localCuboid = zeros(343, 9^3);
% for i = 1:size(localCuboid, 1)
% %     sampleCell = getSurroundCuboid(...
% %         image3d, [x(i), y(i), z(i)], [wSize, wSize, wSize]);
%     sampleCell = getSurroundCuboid(...
%         image3d, [x(i), y(i), z(i)], [9, 9, 9]);
%     localCuboid(i, :) = sampleCell(:)';
% end
% localCuboid = (projMat*localCuboid')';
D = dist2(localCuboid, clusters);
[~, nearest] = min(D, [], 2);
bins = 1:size(clusters, 1);
histogram = histc(nearest', bins);
end
