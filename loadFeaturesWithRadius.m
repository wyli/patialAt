function [features, y] = ...
        loadFeaturesWithRadius(setpath, ind, radius, clicks, locFlag)


    %fprintf('radius:%d, clicks:%d, locFlag:%d\n',...
    %    radius, clicks, locFlag);
    if radius > -1
        filesetname = sprintf([setpath '/*_%d.mat'], radius);
    else % clicked patches
        filesetname = sprintf([setpath '/*.mat']);
    end

    feaHigh = [];
    yHigh = [];
    info = [];
    d = [];

    lsFiles = dir(filesetname);

    % location based affinity
    if radius > -1 && locFlag > 0
        for i = 1:length(ind)

            %fprintf([setpath '/%s\n'], lsFiles(ind(i)).name);
            load([setpath, '/', lsFiles(ind(i)).name]);
            selection = locRef <= clicks;
            feaHigh = [feaHigh, X_features(:, selection)];
            yHigh = [yHigh, info(selection)];
            %imageInd = [imageInd; ones(size(yHigh))' * i];
            %referInd = [referInd; locRef(selection)];
            d = [d; locDist(selection)];
        end
    end

    % feature based affinity
    if radius > -1 && locFlag < 0
        for i = 1:length(ind)

            %fprintf([setpath '/%s\n'], lsFiles(ind(i)).name);
            load([setpath, '/', lsFiles(ind(i)).name]);
            selection = feaRef <= clicks;
            feaHigh = [feaHigh, X_features(:, selection)];
            yHigh = [yHigh, info(selection)];
            %imageInd = [imageInd; ones(size(yHigh))' * i];
            %referInd = [referInd; feaRef(selection)];
            d = [d; feaDist(selection)];
        end
    end

    % clicked patches
    if radius < 0
        for i = 1:length(ind)

            %fprintf([setpath '/%s\n'], lsFiles(ind(i)).name);
            load([setpath, '/', lsFiles(ind(i)).name]);
            feaHigh = [feaHigh, X_features(:, 1:clicks)];
            yHigh = [yHigh, info(1, 1:clicks)];
        end
    end

    % define patch labels
    features = feaHigh';
    y = zeros(size(yHigh))';

    % labels: location based affinity
    if radius > -1 && locFlag > 0
        yDist = exp(-d/5e7); % change this delta
        for i = 1:length(yHigh)

            if strcmp(yHigh{i}.type, 'LGD')
                y(i) = -yDist(i);
            else
                y(i) = yDist(i);
            end
        end
    end

    % labels: feature based affinity
    if radius > -1 && locFlag < 0
        yDist = exp(-d/25000); % change this sigma
        for i = 1:length(yHigh)

            if strcmp(yHigh{i}.type, 'LGD')
                y(i) = -yDist(i);
            else
                y(i) = yDist(i);
            end
        end
    end

    % labels: clicked patches (binary label)
    if radius < 0
        for i = 1:length(yHigh)

            if strcmp(yHigh{i}.type, 'LGD')
                y(i) = -0.5;
            else
                y(i) = 0.5;
            end
        end
    end
end
