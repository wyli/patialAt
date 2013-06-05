diary off;
diary('~/output/com-15.log');
%parfor schemeInd = 1:10
for schemeInd = 1:10
    for click = 15
        expFactory(schemeInd, click, 'prop-huber-fea', 1);
        expFactory(schemeInd, click, 'prop-squ-fea', 1);
        expFactory(schemeInd, click, 'prop-huber-loc', 1);
        expFactory(schemeInd, click, 'prop-squ-loc', 1);
    end
end
%parfor schemeInd = 1:10
%    for click = 6:8
%        expFactory(schemeInd, click, 'prop-huber-fea', 1);
%        expFactory(schemeInd, click, 'prop-squ-fea', 1);
%        expFactory(schemeInd, click, 'prop-huber-loc', 1);
%        expFactory(schemeInd, click, 'prop-squ-loc', 1);
%    end
%end
%parfor schemeInd = 1:10
%    for click = 9
%        expFactory(schemeInd, click, 'prop-huber-fea', 1);
%        expFactory(schemeInd, click, 'prop-squ-fea', 1);
%        expFactory(schemeInd, click, 'prop-huber-loc', 1);
%        expFactory(schemeInd, click, 'prop-squ-loc', 1);
%    end
%end
%parfor schemeInd = 1:10
%    for click = 10
%        expFactory(schemeInd, click, 'prop-huber-fea', 1);
%        expFactory(schemeInd, click, 'prop-squ-fea', 1);
%        expFactory(schemeInd, click, 'prop-huber-loc', 1);
%        expFactory(schemeInd, click, 'prop-squ-loc', 1);
%    end
%end
diary off;
