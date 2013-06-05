for i = -40:-1
fprintf('model selecting... ');
trainYHigh = double(trainYHigh);
feaHigh = double(feaHigh);
if strfind(model, 'prop-squ-fea')
    % proposed-squ-fea
    log10c = -5; log10p = -9;
    cmd = ['-s 0 -c ', num2str(10^log10c), ' -p ', num2str(10^log10p)];
elseif strfind(model, 'prop-squ-loc')
    % proposed-squ-loc
    log10c = -5; log10p = -9;
    cmd = ['-s 0 -c ', num2str(10^log10c), ' -p ', num2str(10^log10p)];
elseif strfind(model, 'prop-huber-fea')
    %proposed-huber-fea
    log2c = 3; log2p = i;
    cmd = ['-s 2 -c ', num2str(2^log2c), ' -p ', num2str(2^log2p)];
    %cmd = ['-s 2 -c 1214 -p ', num2str(10^log10p)];
elseif strfind(model, 'prop-huber-loc')
    %proposed-huber-fea
    log2c = 3; log2p = -4;
    cmd = ['-s 2 -c ', num2str(2^log2c), ' -p ', num2str(2^log2p)];
    %cmd = ['-s 2 -c 1214 -p ', num2str(10^log10p)];
end

%modelbest = train(sparse(trainYHigh), sparse(feaHigh), [cmd ' -q']);
modelbest = train(sparse(trainYHigh), sparse(feaHigh), cmd);
fprintf('cmd: %s\n', cmd);

%test
testY = double(testY > 0) * 2 - 1; % testing binary labels
feaTest = double(feaTest);
[predicted, testacc, testscores] = predict(...
    sparse(testY), sparse(feaTest), modelbest, ' -q');
scores(isnan(testscores)) = 0;
if modelbest.Label(1) == -1
    testscores = -testscores;
end
[~, ~, ~, testauc] = perfcurve(testY, testscores, 1);
fprintf('test auc: %.3f\n', testauc);
%save(filename, 'predicted', 'testauc', 'testscores', 'testY', 'testauc');
end
