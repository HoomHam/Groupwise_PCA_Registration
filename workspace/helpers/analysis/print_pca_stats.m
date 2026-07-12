input_dir      = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/';
comparison_dir = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/';
threshold = 0.04; n_components = 3; vote_thresh = 9;

rspace_files = dir(fullfile(input_dir, 'rspace*.mat'));
mat_raw = load(fullfile(input_dir, rspace_files(1).name), 'image');
original = double(mat_raw.image); original = original / max(original(:));
vm = sum(original >= threshold, 4) >= vote_thresh;
err_orig = computePCAError(original, n_components); err_orig = err_orig .* vm;

runs = { ...
    'original',               'original'; ...
    'ants_image_only',        'ANTs  img-only'; ...
    'ants_enhance_nonpad',    'ANTs  enhance-np'; ...
    'ants_enhance_pad',       'ANTs  enhance-p'; ...
    'ants_denoise_nonpad',    'ANTs  denoise-np'; ...
    'ants_denoise_pad',       'ANTs  denoise-p'; ...
    'ants_clahe_nonpad',      'ANTs  clahe-np'; ...
    'ants_clahe_pad',         'ANTs  clahe-p'; ...
    'elastix_image_only',     'Elx   img-only'; ...
    'elastix_enhance_nonpad', 'Elx   enhance-np'; ...
    'elastix_enhance_pad',    'Elx   enhance-p'; ...
    'elastix_denoise_nonpad', 'Elx   denoise-np'; ...
    'elastix_denoise_pad',    'Elx   denoise-p'; ...
    'elastix_clahe_nonpad',   'Elx   clahe-np'; ...
    'elastix_clahe_pad',      'Elx   clahe-p'; ...
};

meds = zeros(size(runs,1),1);
meds(1) = median(err_orig(vm));
for r = 2:size(runs,1)
    p = fullfile(comparison_dir, runs{r,1}, 'gas.mat');
    s = load(p,'gas'); img = double(s.gas)/max(double(s.gas(:)));
    e = computePCAError(img, n_components) .* vm;
    meds(r) = median(e(vm));
end

ants_base = meds(2);
elx_base  = meds(9);

fprintf('\n%-22s  median_err    vs_own_img_base\n', 'run');
fprintf('%s\n', repmat('-',1,52));
for r = 1:size(runs,1)
    if r <= 8
        base = ants_base;
    else
        base = elx_base;
    end
    pct = (base - meds(r)) / base * 100;
    fprintf('%-22s  %9.5f    %+7.2f%%\n', runs{r,2}, meds(r), pct);
    if r == 1 || r == 8
        fprintf('%s\n', repmat('-',1,52));
    end
end
