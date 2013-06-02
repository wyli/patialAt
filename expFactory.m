function expFactory(schemeInd, clicks, model, repeating)
%
% choice of model:
% 'svm-ign',
% 'svm-all-fea', 'prop-huber-fea', 'prop-squ-fea'
% 'svm-all-loc', 'prop-huber-loc', 'prop-squ-loc'
%

% start of test case
%schemeInd = 1;
%clicks = 50;
%model = 'svm-ign';
%model = 'svm-all-fea';
%model = 'svm-all-loc';
%model = 'prop-huber-fea';
%repeating = 1;
% end of test case

fprintf('*** start experiments ***\n');
fprintf('at: %s\n', datestr(now));
s = RandStream('mt19937ar','Seed','shuffle');
if strfind(model, 'loc')
    locFlag = 1;
else
    locFlag = -1;
end
fprintf('model: %s, loc-based: %d, fold: %d, clicks: %d, repeating: %d\n',...
    model, locFlag, schemeInd, clicks, repeating);

dirString = '~/output/exp_20130529T185037_primal/result_%02d/fea%d';
currentDir = sprintf(dirString, schemeInd, clicks);
cd(currentDir);

filename = sprintf('%s_%d_%d_%02d', model, schemeInd, clicks, repeating);

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
%lowImageInd = [];
%referLowInd = [];
if ~strcmp(model, 'svm-ign')
    for r = [0, 5, 20, 25, 40, 50, 80, 100]
        %[tempfeaLow, temptrainYLow, templowImageInd, tempreferLowInd] = ...
        %    loadFeaturesWithRadius('./low', trainInd, r, clicks, locFlag);
        [tempfeaLow, temptrainYLow] = ...
            loadFeaturesWithRadius('./low', trainInd, r, clicks, locFlag);
        r = randsample(s, size(tempfeaLow,1), ceil(.2 * size(tempfeaLow,1)));
        %r = randsample(s, size(tempfeaLow,1), size(tempfeaLow,1));
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

if ~isempty(feaLow) && ~isempty(trainYLow)
    feaHigh = [feaHigh; feaLow];
    trainYHigh = [trainYHigh; trainYLow];
end

if strfind(model, 'svm')
    aucbest = 0;
    bestcmd = [];
    modelbest = [];

    % train
    fprintf('model selecting... ');
    trainYHigh = double(trainYHigh > 0) * 2 - 1; % binary labels.
    feaHigh = double(feaHigh);
    for log10c = -1:-1:-2
        cmd = ['-s 2 -c ', num2str(10^log10c)];
        modelnow = train(sparse(trainYHigh), sparse(feaHigh), [cmd ' -q']);
        [~, ~, scores] = predict(...
            sparse(trainYHigh), sparse(feaHigh), modelnow, ' -q');

        scores(isnan(scores)) = 0;
        if modelnow.Label(1) == -1
            scores = -scores;
        end

        [~, ~, ~, auc] = perfcurve(trainYHigh, scores, '1');
        if (auc > aucbest && sum(modelnow.w) ~= 0)
            bestcmd = cmd;
            aucbest = auc;
            modelbest = modelnow;
        end
    end
    fprintf('cmd: %s\n', bestcmd);
    fprintf('train auc: %.3f\n', aucbest);

    %test
    fprintf('model testing... ');
    testY = double(testY > 0) * 2 - 1; % binary labels
    feaTest = double(feaTest);
    [predicted, testacc, testscores] = predict(...
        sparse(testY), sparse(feaTest), modelbest, ' -q');
    scores(isnan(testscores)) = 0;
    if modelnow.Label(1) == -1
        testscores = -testscores;
    end
    [~, ~, ~, testauc] = perfcurve(testY, testscores, 1);
    fprintf('test auc: %.3f\n', testauc);
    %save(filename, 'predicted', 'testacc', 'testscores', 'testY', 'testauc');
end

fprintf('%s: loc-based: %d, fold: %d, clicks: %d, repeating: %d, %.3f\n',...
    model, locFlag, schemeInd, clicks, repeating, testauc);
fprintf('at: %s\n', datestr(now));
fprintf('*** end experiments ***\n');
