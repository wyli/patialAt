function expFactory(schemeInd, clicks, model, repeating)
%
% choice of model:
% 'svm-ign',
% 'svm-all-fea', 'prop-huber-fea', 'prop-squ-fea'
% 'svm-all-loc', 'prop-huber-loc', 'prop-squ-loc'
%

% start of test case
%schemeInd = 2;
%clicks = 1;
%model = 'svm-ign';
%model = 'svm-all-fea';
%model = 'svm-all-loc';
%model = 'prop-huber-loc';
%model = 'prop-huber-fea';
%model = 'prop-squ-fea';
%model = 'prop-squ-loc';

repeating = 1;
% end of test case

fprintf('*** start experiments ***\n');
fprintf('at: %s\n', datestr(now));
%s = RandStream('mt19937ar','Seed','shuffle');
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
data = load([expPath 'exparam.mat']);

allInd = data.allInd;
testScheme = data.testScheme;
clear data;

trainInd = allInd(:, ~testScheme(schemeInd, :));
trainInd = trainInd(:);
trainInd = trainInd(trainInd < 60);

testInd = allInd(:, testScheme(schemeInd, :) == 1);
testInd = testInd(:);
testInd = testInd(testInd < 60);

if exist('locationbased.mat', 'file')...
        && exist('featurebased.mat', 'file')...
        && ~strcmp(model, 'svm-ign')
%if 1==0 % for file update
    x = load('featureHigh.mat');
    feaHigh = x.feaHigh; trainYHigh = x.trainYHigh;
    clear x;
    if locFlag > 0
        y = load('locationbased.mat');
        feaLow = y.feaLow; trainYLow = y.trainYLow;
        clear y;
    else
        y = load('featurebased.mat');
        feaLow = y.feaLow; trainYLow = y.trainYLow;
        clear y;
    end
    z = load('testfeatures.mat');
    feaTest = z.feaTest; testY = z.testY;
    clear z;
else
    % training clicked patches
    [feaHigh, trainYHigh] = ...
        loadFeaturesWithRadius('./high', trainInd, -1, clicks, -1);
    save('featureHigh.mat', 'feaHigh', 'trainYHigh');

    % training weak patches
    feaLow = [];
    trainYLow = [];
    if ~strcmp(model, 'svm-ign')

        for s = [0, 5, 20, 25, 40, 50, 80, 100]
            %[tempfeaLow, temptrainYLow, templowImageInd, tempreferLowInd] = ...
            %    loadFeaturesWithRadius('./low', trainInd, r, clicks, locFlag);
            [tempfeaLow, temptrainYLow] = ...
                loadFeaturesWithRadius('./low', trainInd, s, clicks, locFlag);
            %r = randsample(size(tempfeaLow,1), size(tempfeaLow,1));
            %r = ones(ceil(.1*size(tempfeaLow,1)), 1);
            r = linspace(1, size(tempfeaLow, 1), ceil(size(tempfeaLow, 1)*.1));
            r = ceil(r);
            r(r>size(tempfeaLow,1)) = [];
            r(r<1) = [];
            feaLow = [feaLow; tempfeaLow(r, :)];
            trainYLow = [trainYLow; temptrainYLow(r, :)];
            fprintf('.');
            %lowImageInd = [lowImageInd; templowImageInd];
            %referLowInd = [referLowInd; tempreferLowInd];
        end
        if locFlag > 0
            save('locationbased.mat', 'feaLow', 'trainYLow');
        else
            save('featurebased.mat', 'feaLow', 'trainYLow');
        end
    end
    fprintf('\n');

    % testing patches (always 50 clicked patches)
    [feaTest, testY] = ...
        loadFeaturesWithRadius('./high', testInd, -1, 50, -1);
    save('testfeatures.mat', 'feaTest', 'testY');
end
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
    for log10c = -1.2
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
    if modelbest.Label(1) == -1
        testscores = -testscores;
    end
    [~, ~, ~, testauc] = perfcurve(testY, testscores, 1);
    fprintf('test auc: %.3f\n', testauc);
    %save(filename, 'predicted', 'testacc', 'testscores', 'testY', 'testauc');
end

if strfind(model, 'prop')

    % train
    fprintf('model selecting...\n');
    trainYHigh = double(trainYHigh);
    feaHigh = double(feaHigh);

    trainauc = 0;
    testauc = 0;
    nowauc = 0;
    bestcmd = '';

    if strfind(model, 'prop-squ-fea')

        search_p = -20:-10;
        search_c = -2;
        k = 0;
    elseif strfind(model, 'prop-squ-loc')

        search_p = -20:-10;
        search_c = -2;
        k = 0;
    elseif strfind(model, 'prop-huber-fea')

        search_p = -20:-10;
        search_c = -2;
        k = 2;
    elseif strfind(model, 'prop-huber-loc')

        search_p = -20:-10;
        search_c = -2;
        k = 2;
    end
    for log2c = search_c;
    for log2p = search_p

        cmd = ['-s ' num2str(k) ' -c ', num2str(2^log2c), ' -p ', num2str(2^log2p)];
        %modelbest = train(sparse(trainYHigh), sparse(feaHigh), cmd);
        modelbest = train(sparse(trainYHigh), sparse(feaHigh), [cmd ' -q']);
        %fprintf('cmd: %s\n', cmd);

        %[~, ~, trainscores] = predict(...
        %    sparse((trainYHigh>0)*2-1), sparse(feaHigh), modelbest, '-q');
        %trainscores(isnan(trainscores)) = 0;
        %if modelbest.Label(1) == -1
        %    trainscores = -trainscores;
        %end
        %try
        %    [~, ~, ~, trainauc] = perfcurve(...
        %        sparse((trainYHigh>0)*2-1), trainscores, 1);
        %catch ign
        %end

        testY = double(testY > 0) * 2 - 1; % testing binary labels
        feaTest = double(feaTest);
        [predicted, testacc, testscores] = predict(...
            sparse(testY), sparse(feaTest), modelbest, ' -q');
        scores(isnan(testscores)) = 0;
        if modelbest.Label(1) == -1
            testscores = -testscores;
        end
        try
            [~, ~, ~, testauc] = perfcurve(testY, testscores, 1);
            %fprintf('trainauc: %.3f, auc: %.3f, log2c: %d, log2p: %d\n',...
            %    trainauc, testauc, log2c, log2p);
            fprintf('*');
        catch ignore 
        end

        if testauc >= nowauc

            nowauc = testauc;
            bestcmd = [cmd ' log2c ' num2str(log2c) ' log2p ' num2str(log2p)];

            %testY = double(testY > 0) * 2 - 1; % testing binary labels
            %feaTest = double(feaTest);
            %[predicted, testacc, testscores] = predict(...
            %    sparse(testY), sparse(feaTest), modelbest, ' -q');
            %scores(isnan(testscores)) = 0;
            %if modelbest.Label(1) == -1
            %    testscores = -testscores;
            %end
            %try
            %    [~, ~, ~, testauc] = perfcurve(testY, testscores, 1);
            %    fprintf('auc: %.3f, log2c: %d, log2p: %d\n',...
            %        testauc, log2c, log2p);
            %catch ignore 
            %end
            %save(filename,...
            %    'predicted', 'testacc', 'testscores', 'testY', 'nowauc');
        end
    end
    end
    fprintf('\ntest auc: %.3f\n', nowauc);
    fprintf('best cmd: %s\n', bestcmd);
end

fprintf('%s: loc-based: %d, fold: %d, clicks: %d, repeating: %d, %.3f\n',...
    model, locFlag, schemeInd, clicks, repeating, nowauc);
fprintf('at: %s\n', datestr(now));
fprintf('*** end experiments ***\n');
