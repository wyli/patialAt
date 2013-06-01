addpath(genpath('~/documents/opt_learning/randomfeatures/'));
parfor i = 2:10
    for j = 1:50
        findreference(i, j);
    end
end
