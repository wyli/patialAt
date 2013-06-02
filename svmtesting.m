diary('~/output/svmall.log');
parfor schemeInd = 1:10
    for click = 1:50
        %expFactory(schemeInd, click, 'svm-ign', 1);
        expFactory(schemeInd, click, 'svm-all-fea', 1);
        expFactory(schemeInd, click, 'svm-all-loc', 1);
    end
end
diary off;
