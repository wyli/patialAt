function scaled = scaleFeatures(data)
% each feature is a row vector

scaled = data - mean(data, 2)* ones(1, size(data, 2));
scaled = scaled ./(std(scaled, 0, 2) * ones(1, size(scaled, 2)));
scaled(isnan(scaled)) = 0;
end