function  [cuboid_low, cuboid_high] = img2Cub(...
    imgFile, segFile, windowSize, step, radius)

numOfSamples = 50;
numOfWeakSamples = 500;
% load images and segmentations.
load(imgFile);
load(segFile);

% get interesting locations
[~, locations3d] = scanForPositiveSampleLocations(...
    segImg, windowSize, step);
fprintf('Found %d all locations\n', length(locations3d));
randIndex = randsample(size(locations3d,1), min(size(locations3d,1), numOfSamples));
fprintf('Randomly choose high confident locations: %d\n ',...
            min(size(locations3d,1), numOfSamples));

locations3d = locations3d(randIndex, :);
cuboid_high = cell(2, size(randIndex,1));
for loc = 1:size(randIndex,1)
    cuboid_high{1,loc} = getSurroundCuboid(...
        oriImg, locations3d(loc,:), windowSize);
    cuboid_high{2,loc} = locations3d(loc,:);
end	

fprintf('propagate to low confidence: %d\n', min(size(locations3d,1),numOfWeakSamples));
[~, locations3d] = scanForWeakSampleLocations(...
    segImg, windowSize, step, radius);
fprintf('Found %d all weak locations\n', length(locations3d));
randIndex = randsample(size(locations3d,1), min(size(locations3d,1),numOfWeakSamples));
fprintf('Randomly choose low confident locations: %d\n ',...
            min(size(locations3d,1), numOfWeakSamples));

locations3d = locations3d(randIndex, :);
cuboid_low = cell(2, size(randIndex,1));
for loc = 1:size(randIndex,1)
    cuboid_low{1,loc} = getSurroundCuboid(...
        oriImg, locations3d(loc,:), windowSize);
    cuboid_low{2,loc} = locations3d(loc,:);
end	
end % end of function
