function [scaled, scaleVectors] = scaleFeatures(data, scalVectors, testFlag)
if testFlag < 0
    mind = min(data, [], 1);
    maxd = max(data, [], 1);
    scaleVectors = [mind; maxd];
end
scaled = (data - repmat(scaleVectors(1, :),size(data,1),1))...
    *spdiags(1./(scaleVectors(2, :)-scaleVectors(1, :))',0,size(data,2),size(data,2));
end
