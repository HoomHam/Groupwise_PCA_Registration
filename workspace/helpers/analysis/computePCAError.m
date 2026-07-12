function [pcaErrorMap, score, coeff, latent, explained] = computePCAError(img4D, nComponents)

    % some_threshold = 0.03;

    % img4D: 4D image (X x Y x Z x T)
    % nComponents: number of principal components to use for reconstruction
    [nx, ny, nz, nt] = size(img4D);
    N = nx * ny * nz;
    
    % Reshape to (n_voxels x T)
    dataMatrix = reshape(img4D, [N, nt]);
    
    % Perform PCA (you can use SVD or a PCA function)
    [coeff, score, latent, tsquared, explained, mu] = pca(dataMatrix);
    
    % Reconstruct the data using the top nComponents
    reconstruction = score(:, 1:nComponents) * coeff(:, 1:nComponents)';
    
    
    % Compute reconstruction error per voxel (sum of squared differences)
    errorPerVoxel = sum((dataMatrix - reconstruction).^2, 2);
    
    % Put the error back into the 3D space (assuming no masking was applied)
    pcaErrorMap = reshape(errorPerVoxel, [nx, ny, nz]);

    % Optionally, mask out background voxels:
    % mask = reshape((mean(dataMatrix, 2) > some_threshold),[size(img4D,1) size(img4D,2) size(img4D,3)]);
    
    % pcaErrorMap(mask==0) = NaN;
    
end