function [] = drawSamples(imgSet, xmlSet, outputSet, windowSize)
% usage: 
% >> drawSamples('~/desktop/OPTmix', '~/desktop/description', '~/desktop/cuboidset', 21);
fprintf('%s drawing samples, (may contain empty cell.\n', datestr(now));
fprintf('windowSize: %d\n', windowSize);
% input
xmlFiles = dir([xmlSet '/*.xml']);
segImgSet = '%s/Annotated/%s%s';
oriImgSet = '%s/Images/%s%s';
% output
outputSet = sprintf('%s/cuboid_%d', outputSet, windowSize);
fprintf('output: %s\n', outputSet);
mkdir(outputSet);
mkdir([outputSet '/low']);
mkdir([outputSet '/high']);
% parameters
window3d = windowSize * ones(1,3);
step3d = window3d;

for i = 1:size(xmlFiles, 1)
    cuboid_high = {};
    cuboid_low = {};
    rec = VOCreadxml([xmlSet '/' xmlFiles(i).name]);
    name = rec.annotation.index;
    for p = 1:size(rec.annotation.part, 2)
        part = rec.annotation.part{p};
        segFile = sprintf(segImgSet, imgSet, name, part);
        oriFile = sprintf(oriImgSet, imgSet, name, part);
        fprintf('input: %s\n', segFile);
        fprintf('input: %s\n', oriFile);
        [cubPart_low, cubPart_high] = img2Cub(oriFile, segFile, window3d, step3d);
        cuboid_high = [cuboid_high, cubPart_high];
        cuboid_low = [cuboid_low, cubPart_low];
    end
    for j = 1:size(cuboid_high, 2)
        cuboid_high{3, j} = rec.annotation;
    end
    for j = 1:size(cuboid_low, 2)
        cuboid_low{3, j} = rec.annotation;
    end
    badIndex = cellfun(@isempty, cuboid_low(1, :));
    cuboid_low(:, badIndex) = [];
    cuboid = cuboid_low;
    cuboidSet = sprintf('%s/low/%s', outputSet, name);
    fprintf('saving at: %s\n', cuboidSet);
    save(cuboidSet, 'cuboid');

    badIndex = cellfun(@isempty, cuboid_high(1, :));
    cuboid_high(:, badIndex) = [];
    cuboid = cuboid_high;
    cuboidSet = sprintf('%s/high/%s', outputSet, name);
    fprintf('saving at: %s\n', cuboidSet);
    save(cuboidSet, 'cuboid');

    clear cuboid_low cuboid_high;
end
end % end of function
