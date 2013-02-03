clear all; close all;
% image paths
xmlSet = '~/desktop/description';
patchesSet = '~/desktop/Cuboid_21/';
outputFolder = '~/desktop/cuboidset/';

xmlFiles = dir([xmlSet '/*.xml']);
for i = 1:size(xmlFiles)
    rec = VOCreadxml([xmlSet '/' xmlFiles(i).name]);
    load([patchesSet, rec.annotation.index]);
    fprintf('Reading: %s, patches: %d\n', rec.annotation.index, size(cuboid, 2));
    for j = 1:size(cuboid, 2)
        cuboid{3, j} = rec.annotation;
    end
    save([outputFolder, rec.annotation.index], 'cuboid');
end
