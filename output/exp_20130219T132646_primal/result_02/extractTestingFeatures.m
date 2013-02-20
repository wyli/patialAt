mkdir('testingfea');
addpath(genpath('~/documents/opt_learning/randomfeatures'));

load('randMat.mat');
samplePerFile = 100000;
radius = -1;

cuboidInput = '~/desktop/cuboidset/cuboid_%d/high/%s';
feaOutput = 'testingfea';

extractFeatures(...
    '.', cuboidInput, feaOutput,...
    2, 9, 1, randMat, samplePerFile, radius);
