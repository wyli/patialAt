clear all; close all;
RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', sum(100*clock)));
addpath('~/dropbox/libr/matlab/');
addpath(genpath('~/documents/opt_learning/randomfeatures'));

id = '';
if isempty(id)
    id = datestr(now, 30);
    id = sprintf('%s_primal', id);
end
out_dir = '~/documents/opt_learning/randomfeatures/output/';
out_dir = sprintf('%sexp_%s', out_dir, id);
mkdir(out_dir);
diary off;
diary([out_dir '/exp.log']);

% input patches
patchSet = '~/desktop/cuboidset';

% flags
generate_scheme = 1;
do_kmeans = ones(3, 1);
do_extract_features= 1;
do_classification = 1;
fprintf('at: %s\n', datestr(now));
fprintf('generate testing scheme? %d\n', generate_scheme);
fprintf('%s: \n', 'I will');
fprintf('%s: %d\n', 'do kmeans on training', do_kmeans(1));
fprintf('%s: %d\n', 'extract features on all', do_extract_features);
fprintf('%s: %d\n', 'do classification', do_classification);
fprintf('\n\n');
% end of flags

windowSize = 21;
subWindow = 9;
if generate_scheme
    k = 6;
    foldSize = 10;
    allInd = randsample(k*foldSize, k*foldSize);
    allInd = reshape(allInd, foldSize, []);
    testScheme = eye(k, 'int8');
    save([out_dir '/exparam'], 'testScheme', 'allInd');
else
    load([out_dir '/exparam']);
end

repeating = 3;
for f = 1:min(repeating, length(testScheme))

    % output dir
    resultSet = sprintf('%s/result_%02d', out_dir, f);
    mkdir(resultSet);

    trainInd = allInd(:, ~testScheme(f, :));
    trainInd = trainInd(:);
    trainInd = trainInd(trainInd~=60);
    testInd = allInd(:, f);
    testInd = testInd(testInd~=60); % we only have 59 files
    fprintf('training on:\n');
    for i = trainInd
        fprintf('%d, ', i);
    end
    fprintf('\ntesting on:\n');
    for j = testInd
        fprintf('%d, ', j);
    end
    fprintf('\n');

    if do_kmeans(f)

        randMat = randn(150, 9^3);
        save([resultSet, '/randMat'], 'randMat');
        kcenters = 200;
        samplePerFile = 50; % use all key points.
        trainBases(...
            resultSet, patchSet, trainInd,...
            windowSize, subWindow, 2, kcenters, randMat, samplePerFile)
    end

    if do_extract_features

        if ~do_kmeans(f)
            load([resultSet, '/randMat']);
        end
        mkdir([resultSet, '/fea']);
        cuboidInput = [patchSet, '/cuboid_%d/high/%s'];
        feaOutput = [resultSet, '/fea/high'];
        samplePerFile = 50;
        extractFeatures(...
            resultSet, cuboidInput, feaOutput,...
            windowSize, 9, 1, randMat, samplePerFile, -1);

        cuboidInput = [patchSet, '/cuboid_%d/low/%s'];
        feaOutput = [resultSet, '/fea/low'];
        samplePerFile = 500;
        radius = 0;
        extractFeatures(...
            resultSet, cuboidInput, feaOutput,...
            windowSize, 9, 1, randMat, samplePerFile, radius);
    end

    if do_classification
    end
end

diary off;
