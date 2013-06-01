function expFactory(schemeInd, clicks, model)
%
% choice of model:
% 'svm-ign',
% 'svm-all-fea', 'prop-huber-fea', 'prop-squ-fea'
% 'svm-all-loc', 'prop-huber-loc', 'prop-squ-loc'
%

% start of test case
%schemeInd = 1;
%clicks = 1;
%model = 'svm-ign';
%model = 'prop-huber-fea';
% end of test case
fprintf('*** start experiments ***\n');
if strfind(model, 'loc')
    locFlag = 1;
else
    locFlag = -1;
end
fprintf('model: %s, location-based: %d, clicks: %d\n', model, locFlag, clicks);

dirString = '~/output/exp_20130529T185037_primal/result_%02d/fea%d';
currentDir = sprintf(dirString, schemeInd, clicks);
cd(currentDir);
fprintf([pwd '\n']);

addpath('~/documents/opt_learning/randomfeatures/');
addpath(genpath('~/documents/opt_learning/randomfeatures/util/'));
addpath(genpath('~/documents/opt_learning/randomfeatures/kmeans2/'));
warning('off', 'MATLAB:rmpath:DirNotFound');
if strfind(model, 'svm')
    fprintf('plain SVM\n');
    addpath(genpath('~/desktop/liblinear-1.93/matlab'));
    rmpath(genpath('~/dropbox/libr/matlab'));
else
    fprintf('proposed work\n');
    addpath(genpath('~/dropbox/libr/matlab'));
    rmpath(genpath('~/desktop/liblinear-1.93/matlab'));
end
which train
expPath = '/Users/wenqili/output/exp_20130529T185037_primal/';
load([expPath 'exparam.mat']);

trainInd = allInd(:, ~testScheme(schemeInd, :));
trainInd = trainInd(:);
trainInd = trainInd(trainInd < 60);

testInd = allInd(:, testScheme(schemeInd, :) == 1);
testInd = testInd(:);
testInd = testInd(testInd < 60);

% training clicked patches
[feaHigh, trainYHigh] = ...
    loadFeaturesWithRadius('./high', trainInd, -1, clicks, -1);

% training weak patches
feaLow = [];
trainYLow = [];
lowImageInd = [];
referLowInd = [];
if ~strcmp(model, 'svm-ign')
    for r = [0, 5, 20, 25, 40, 50, 80, 100]
        %[tempfeaLow, temptrainYLow, templowImageInd, tempreferLowInd] = ...
        %    loadFeaturesWithRadius('./low', trainInd, r, clicks, locFlag);
        [tempfeaLow, temptrainYLow] = ...
            loadFeaturesWithRadius('./low', trainInd, r, clicks, locFlag);
        r = randsample(size(tempfeaLow,1), ceil(.2 * size(tempfeaLow,1)));
        feaLow = [feaLow; tempfeaLow(r, :)];
        trainYLow = [trainYLow; temptrainYLow(r, :)];
        fprintf('.');
        %lowImageInd = [lowImageInd; templowImageInd];
        %referLowInd = [referLowInd; tempreferLowInd];
    end
end
fprintf('\n');

% testing patches (always 50 clicked patches)
[feaTest, testY] = ...
    loadFeaturesWithRadius('./high', testInd, -1, 50, -1);
fprintf('size clicked features: %dx%d\n', size(feaHigh,1), size(feaHigh,2));
fprintf('size weak features: %dx%d\n', size(feaLow,1), size(feaLow,2));
fprintf('size testing features: %dx%d\n', size(feaTest,1), size(feaTest,2));
fprintf('key-weak ratio: %.3f\n', size(feaLow, 1)/size(feaHigh, 1));
