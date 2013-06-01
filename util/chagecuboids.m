% script add one field to cuboids. (flatten dense samples, window size 9, step 2)
%load random matrix first!
lsfiles = dir('*.mat');
for i = 1:length(lsfiles)
    load(lsfiles(i).name);
    fprintf('%s\n', lsfiles(i).name);
    for j = 1:length(cuboid)
        [x, y, z] = meshgrid(5:2:17);
        x = x(:);
        y = y(:);
        z = z(:);
        localCuboid = zeros(343, 9^3);
        for a = 1:size(localCuboid, 1)
            sampleCell = getSurroundCuboid(...
                cuboid{1,j}, [x(a), y(a), z(a)], [9, 9, 9]);
            localCuboid(a, :) = sampleCell(:)';
        end
        localCuboid = (randMat*localCuboid')';
        cuboid{4,j}=single(localCuboid);
    end
    size(cuboid)
    save(lsfiles(i).name, 'cuboid');
end
