listFiles = dir('*.mat');
% 
% 
% high = [];
% yhigh = [];
% for i = 1:length(listFiles)
%     if (i == 16 || i == 14 || i == 26)
%         continue;
%     end
%     load(listFiles(i).name);
%     ind = randsample(size(X_features, 2), 100);
%     high = [high,X_features(:, ind)]; 
%     yhigh = [yhigh, info{ind}];
% end
% 
% 
% testhigh = [];
% yhigh_t = [];
% for i = 1:length(listFiles)
%     if (i == 14 || i == 16 || i == 26)
%     load(listFiles(i).name);
%     ind = randsample(size(X_features, 2), 100);
%     testhigh = [testhigh,X_features(:, ind)]; 
%     yhigh_t = [yhigh_t, info{ind}];
%     end
% end

% low = [];
% ylow = [];
% for i = 1:length(listFiles)
%     if (i == 16 || i == 14 || i == 26)
%         continue;
%     end
%     load(listFiles(i).name);
%     ind = randsample(size(X_features, 2), 200);
%     low = [low,X_features(:, ind)]; 
%     ylow = [ylow, info{ind}];
% end
% 
% testlow = [];
% ylow_t = [];
% for i = 1:length(listFiles)
%     if (i == 14 || i == 16 || i == 26)
%     load(listFiles(i).name);
%     ind = randsample(size(X_features, 2), 100);
%     testlow = [testlow,X_features(:, ind)]; 
%     ylow_t = [ylow_t, info{ind}];
%     end
% end

% trainHigh = high'; clear high;
% trainLow = low'; clear low;
% testHigh = testhigh'; clear testhigh;
% testLow = testlow'; clear testlow;
% 
% for i = 1:length(yhigh)
%     if strcmp(yhigh(i).type,'LGD')
%         trainYHigh(i) = -2;
%     else
%         trainYHigh(i) = 2;
%     end
% end
% 
% for i = 1:length(ylow)
%     if strcmp(ylow(i).type,'LGD')
%         trainYLow(i) = -0.5;
%     else
%         trainYLow(i) = 0.5;
%     end
% end
% 
% for i = 1:length(yhigh_t)
%     if strcmp(yhigh_t(i).type,'LGD')
%         testYHigh(i) = -2;
%     else
%         testYHigh(i) = 2;
%     end
% end
% 
% for i = 1:length(ylow_t)
%     if strcmp(ylow_t(i).type,'LGD')
%         testYLow(i) = -0.5;
%     else
%         testYLow(i) = 0.5;
%     end
% end
        
% trainYHigh = trainYHigh';
% trainYLow = trainYLow';
% testYHigh = testYHigh';
% testYLow = testYLow';
global mind maxd
%trainall = scaledata([trainHigh; trainLow], 1);
trainall = [trainHigh; trainLow];
trainHigh = [trainHigh, ones(size(trainHigh, 1), 1)];
trainLow = [trainLow, ones(size(trainLow, 1), 1)];
trainall = [trainall, [ones(size(trainHigh, 1),1)*0.01; 0.01*ones(size(trainLow, 1),1)]];
% trainHigh(:, end) = trainHigh(:,end) * 10;
% trainLow(:, end) = trainLow(:,end) * 1;

model = train(sparse([trainYHigh; trainYLow]), sparse(trainall), '-e 0.1 -s 2');
[label, acc, dec]= predict(sparse(testYHigh), sparse(testHigh), model);


model = train(sparse([trainYHigh]), sparse([trainHigh]), '-e 0.01 -s 2');
[label, acc1, dec]= predict(sparse(testYHigh), sparse(testHigh), model);