clear all; close all;
RandStream.setDefaultStream(RandStream('mrg32k3a', 'seed', sum(100*clock)));
addpath('~/dropbox/libprime/matlab/');
addpath('~/documents/opt_learning/randomfeatures/kmeans2/');
id = '';
if isempty(id)
    id = datestr(now, 30);
    id = sprintf('%s_primal', id);
end
out_dir = '~/documents/opt_learning/randomfeatures/output/';
out_dir = sprintf('%s/exp_%s', out_dir, id);
mkdir(out_dir);
diary([out_dir 'exp.log']);

% input patches
patchSet = '~/desktop/cuboidset';

% flags
generate_scheme = 1;
do_kmeans = 1;
do_extract_features= 1;
do_classification = 1;
fprintf('at: %s\n', datestr(now));
fprintf('generate testing scheme? %d', generate_scheme);
fprintf('%s\n', 'I will:');
fprintf('%s: %d\n', 'do kmeans on training', do_kmeans);
fprintf('%s: %d\n', 'extract features on all', do_extract_features);
fprintf('%s: %d\n', 'do classification', do_classification);
fprintf('\n\n');
% end of flags


if generate_scheme
    k = 10;
    foldSize = 3;
    allInd = randsample(30, 30);
    allInd = reshape(allInd, foldSize, []);
    testScheme = eye(k, 'int8');
    save([out_dir '/exparam'], 'testScheme', 'allInd');
else
    load([out_dir '/exparam']);
end

for f = 1:length(testScheme)

    % output dir
    resultSet = sprintf('%s/result_%02d/', out_dir, f);
    mkdir(resultSet);

    trainInd = allInd(:, ~testScheme(f, :));
    trainInd = trainInd(:);
    testInd = allInd(:, f);
    fprintf('training on:\n');
    for i = trainInd
        fprintf('%d, ', i);
    end
    fprintf('\ntesting on:\n');
    for j = testInd
        fprintf('%d, ', i);
    end
    fprintf('\n');

    if do_kmeans

        randMat = randn(150, 9^3);
        save([resultSet, 'randMat'], 'randMat');
        kcenters = 200;
        windowSize = 21;
        subWindow = 9;
        trainBases(...
            resultSet, patchSet, trainInd,...
            windowSize, subWindow, 2, kcenters, randMat)
    end

    if do_extract_features

        if ~do_kmeans
            load([resultSet, 'randMat']);
        end
        cuboidInput = [patchSet, 'cuboid_%d/high/%s'];
        feaOutput = [resultSet, 'fea/high/'];
        extractFeatures(...
            resultSet, cuboidInput, feaOutput,...
            windowSize, subSize, step3d, randMat)

        cuboidInput = [patchSet, 'cuboid_%d/low/%s'];
        feaOutput = [resultSet, 'fea/low/'];
        extractFeatures(...
            resultSet, cuboidInput, feaOutput,...
            windowSize, subSize, step3d, randMat)
    end

    if do_classification
    end
end

diary off;
