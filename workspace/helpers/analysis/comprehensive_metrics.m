% comprehensive_metrics.m
%
% 5 metrics for 15 runs (original + 14 registered):
%   1. PCA reconstruction error  (3 components) — lower = better
%   2. Temporal entropy per voxel               — lower = better
%   3. All-pairs NMI  C(16,2)=120 pairs         — higher = better
%   4. PC1 explained variance %                 — no direction (separate bar)
%   5. Wasserstein dist (orig vs reg per frame) — lower = less data distortion
%
% Outputs:
%   sorted boxplot per metric (4 figures)
%   PC1% bar chart (1 figure)
%   summary heatmap (1 figure)

input_dir      = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/';
comparison_dir = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/';
out_dir        = comparison_dir;

THRESHOLD    = 0.04;
N_COMP       = 3;
VOTE_THRESH  = 9;
NMI_BINS     = 64;
ENT_BINS     = 8;

% {label, color, mask_type, gas_mat_path}
%   color:     k=original  b=ANTs-oneshot  r=Elx-oneshot
%              g=full-pipeline-GW  m=full-pipeline-GW+ANTs
%   mask_type: 'orig' = original EI mask
%              'own'  = compute from this run's own EI frame
c  = @(sub)       fullfile(comparison_dir, sub, 'gas.mat');
gw = @(sub) fullfile(comparison_dir, sub, 'final_groupwise', 'gas.mat');
ga = @(sub) fullfile(comparison_dir, sub, 'final_ants',      'gas.mat');

runs = { ...
    % label                color  mask    path
    'original',            'k',  'orig', ''; ...
    % ── Full pipeline: groupwise only (8 ablation configs) ───────────────────
    'FP-GW full',          'g',  'own',  gw('ablation_500iter_full'); ...
    'FP-GW img-p',         'g',  'own',  gw('ablation_500iter_pad_image_only'); ...
    'FP-GW enh-np',        'g',  'own',  gw('ablation_500iter_npad_enhance'); ...
    'FP-GW enh-p',         'g',  'own',  gw('ablation_500iter_pad_enhance'); ...
    'FP-GW den-np',        'g',  'own',  gw('ablation_500iter_npad_denoise'); ...
    'FP-GW den-p',         'g',  'own',  gw('ablation_500iter_pad_denoise'); ...
    'FP-GW clh-np',        'g',  'own',  gw('ablation_500iter_npad_clahe'); ...
    'FP-GW clh-p',         'g',  'own',  gw('ablation_500iter_pad_clahe'); ...
    % ── Full pipeline: groupwise + ANTs (same 8 configs) ─────────────────────
    'FP-GW+A full',        'm',  'orig', ga('ablation_500iter_full'); ...
    'FP-GW+A img-p',       'm',  'orig', ga('ablation_500iter_pad_image_only'); ...
    'FP-GW+A enh-np',      'm',  'orig', ga('ablation_500iter_npad_enhance'); ...
    'FP-GW+A enh-p',       'm',  'orig', ga('ablation_500iter_pad_enhance'); ...
    'FP-GW+A den-np',      'm',  'orig', ga('ablation_500iter_npad_denoise'); ...
    'FP-GW+A den-p',       'm',  'orig', ga('ablation_500iter_pad_denoise'); ...
    'FP-GW+A clh-np',      'm',  'orig', ga('ablation_500iter_npad_clahe'); ...
    'FP-GW+A clh-p',       'm',  'orig', ga('ablation_500iter_pad_clahe'); ...
    % ── ANTs one-shot (8 configs) ─────────────────────────────────────────────
    'ANTs img',            'b',  'orig', c('ants_image_only'); ...
    'ANTs enh-np',         'b',  'orig', c('ants_enhance_nonpad'); ...
    'ANTs enh-p',          'b',  'orig', c('ants_enhance_pad'); ...
    'ANTs den-np',         'b',  'orig', c('ants_denoise_nonpad'); ...
    'ANTs den-p',          'b',  'orig', c('ants_denoise_pad'); ...
    'ANTs clh-np',         'b',  'orig', c('ants_clahe_nonpad'); ...
    'ANTs clh-p',          'b',  'orig', c('ants_clahe_pad'); ...
    % ── Elastix one-shot (7 configs) ──────────────────────────────────────────
    'Elx img',             'r',  'own',  c('elastix_image_only'); ...
    'Elx enh-np',          'r',  'own',  c('elastix_enhance_nonpad'); ...
    'Elx enh-p',           'r',  'own',  c('elastix_enhance_pad'); ...
    'Elx den-np',          'r',  'own',  c('elastix_denoise_nonpad'); ...
    'Elx den-p',           'r',  'own',  c('elastix_denoise_pad'); ...
    'Elx clh-np',          'r',  'own',  c('elastix_clahe_nonpad'); ...
    'Elx clh-p',           'r',  'own',  c('elastix_clahe_pad'); ...
    % ── Old full-pipeline iteration runs (GW only + GW+ANTs) ─────────────────
    % Iteration schedules: 54320n=[5k,4k,3k,2k] nofilter, 54320=[5k,4k,3k,2k],
    %                      5432=[500,400,300,200], 5000=[5000], 1000=[1000]
    'GW-1k',               'g',  'own',  fullfile(comparison_dir,'1000','gw','gas.mat'); ...
    'GW-5k',               'g',  'own',  fullfile(comparison_dir,'5000','gw','gas.mat'); ...
    'GW-5432',             'g',  'own',  fullfile(comparison_dir,'5432','gw','gas.mat'); ...
    'GW-54320',            'g',  'own',  fullfile(comparison_dir,'54320','gw','gas.mat'); ...
    'GW-54320n',           'g',  'own',  fullfile(comparison_dir,'54320n','gw','gas.mat'); ...
    'GW+A-1k',             'm',  'orig', fullfile(comparison_dir,'1000','ants','gas.mat'); ...
    'GW+A-5k',             'm',  'orig', fullfile(comparison_dir,'5000','ants','gas.mat'); ...
    'GW+A-5432',           'm',  'orig', fullfile(comparison_dir,'5432','ants','gas.mat'); ...
    'GW+A-54320',          'm',  'orig', fullfile(comparison_dir,'54320','ants','gas.mat'); ...
    'GW+A-54320n',         'm',  'orig', fullfile(comparison_dir,'54320n','ants','gas.mat'); ...
};
N_RUNS = size(runs, 1);

% ── Load original ─────────────────────────────────────────────────────────────
rf   = dir(fullfile(input_dir, 'rspace*.mat'));
raw  = load(fullfile(input_dir, rf(1).name), 'image');
orig = double(raw.image);
orig = orig / max(orig(:));

% EI frame of original → largest mask (maximum inflation)
ei_orig    = find_ei_4d(orig);
vm_orig    = orig(:,:,:, ei_orig) >= THRESHOLD;   % EI mask for original + ANTs
fprintf('Original EI frame: %d  |  mask voxels: %d\n', ei_orig, sum(vm_orig(:)));

% ── Storage (cell arrays: each column is a vector of varying length) ──────────
% Metrics that depend on mask size stored as cell columns (variable N_mask per run)
pca_vecs  = cell(N_RUNS, 1);
ent_vecs  = cell(N_RUNS, 1);
% Per-pair NMI: always 120 values regardless of mask (histogram-based, normalized)
N_PAIRS   = nchoosek(16,2);
nmi_vecs  = nan(N_PAIRS, N_RUNS);
% Per-frame Wasserstein: 16 values per run
wass_vecs = nan(16, N_RUNS);
% Scalars
pc1_vals  = nan(N_RUNS, 1);
mask_sizes = nan(N_RUNS, 1);   % track mask voxel count per run

% ── Compute metrics for each run ──────────────────────────────────────────────
pair_idx = nchoosek(1:16, 2);

for r = 1:N_RUNS
    fprintf('[%2d/%d]  %s\n', r, N_RUNS, runs{r,1});

    gas_path = runs{r,4};
    if isempty(gas_path)
        img4d = orig;
    elseif ~isfile(gas_path)
        fprintf('  SKIP — file not found: %s\n', gas_path);
        pca_vecs{r}  = NaN;
        ent_vecs{r}  = NaN;
        mask_sizes(r) = NaN;
        pc1_vals(r)  = NaN;
        continue;
    else
        s = load(gas_path, 'gas');
        img4d = double(s.gas) / max(double(s.gas(:)));
    end

    % ── Assign mask ───────────────────────────────────────────────────────────
    if strcmp(runs{r,3}, 'own')
        ei_run = find_ei_4d(img4d);
        vm_run = img4d(:,:,:, ei_run) >= THRESHOLD;
        fprintf('     EI frame: %d  |  mask: %d vox (own EI)\n', ei_run, sum(vm_run(:)));
    else
        vm_run = vm_orig;
        fprintf('     mask: %d vox (original EI)\n', sum(vm_run(:)));
    end
    mask_sizes(r) = sum(vm_run(:));

    % ── 1. PCA error ──────────────────────────────────────────────────────────
    err_map      = computePCAError(img4d, N_COMP);
    pca_vecs{r}  = err_map(vm_run);   % mean per voxel → size-normalized inherently

    % ── 2. Temporal entropy ───────────────────────────────────────────────────
    ent_vecs{r}  = compute_temporal_entropy(img4d, vm_run, ENT_BINS);  % mean per voxel

    % ── 3. All-pairs NMI ──────────────────────────────────────────────────────
    % NMI is histogram-based (ratio of entropies) → invariant to mask size
    fprintf('     NMI pairs: ');
    for p = 1:N_PAIRS
        A = img4d(:,:,:, pair_idx(p,1));
        B = img4d(:,:,:, pair_idx(p,2));
        nmi_vecs(p,r) = compute_nmi(A, B, vm_run, NMI_BINS);
        if mod(p,24)==0, fprintf('%d ',p); end
    end
    fprintf('\n');

    % ── 4. PC1% ───────────────────────────────────────────────────────────────
    % Explained variance ratio → inherently size-normalized
    pc1_vals(r) = compute_pc1(img4d, vm_run);

    % ── 5. Wasserstein: orig_t vs registered_t ────────────────────────────────
    % Use per-frame intersection mask to avoid boundary voxels that are
    % in-lung in one image but not the other
    % Wasserstein uses quantile function → inherently size-normalized
    for t = 1:16
        orig_t  = orig(:,:,:,t);
        reg_t   = img4d(:,:,:,t);
        vm_t    = vm_run & (orig_t >= THRESHOLD) & (reg_t >= THRESHOLD);
        if sum(vm_t(:)) < 10, continue; end
        wass_vecs(t,r) = wasserstein_1d(orig_t(vm_t), reg_t(vm_t));
    end
end

fprintf('\nAll metrics computed.\n');

% ── Normalization validation ───────────────────────────────────────────────────
fprintf('\nMask size per run (normalization check):\n');
fprintf('%-22s  %8s  %8s  %8s  %8s  %8s\n','run','N_mask','PCA_mean','Ent_mean','NMI_mean','PC1_%');
fprintf('%s\n', repmat('-',1,68));
for r = 1:N_RUNS
    fprintf('%-22s  %8d  %8.5f  %8.5f  %8.5f  %8.2f\n', ...
        runs{r,1}, mask_sizes(r), ...
        mean(pca_vecs{r},'omitnan'), mean(ent_vecs{r},'omitnan'), ...
        mean(nmi_vecs(:,r),'omitnan'), pc1_vals(r));
end

% ── Convert cell vectors to padded matrix for boxplots ────────────────────────
max_mask = max(mask_sizes);
pca_bmat = nan(max_mask, N_RUNS);
ent_bmat = nan(max_mask, N_RUNS);
for r = 1:N_RUNS
    n = numel(pca_vecs{r});
    pca_bmat(1:n, r) = pca_vecs{r};
    ent_bmat(1:n, r) = ent_vecs{r};
end

% ── Scalar summary matrix for heatmap (means) ────────────────────────────────
pca_means  = cellfun(@(v) mean(v,'omitnan'), pca_vecs);
ent_means  = cellfun(@(v) mean(v,'omitnan'), ent_vecs);
nmi_means  = mean(nmi_vecs, 1, 'omitnan')';
wass_means = mean(wass_vecs, 1, 'omitnan')';

metric_means = [pca_means, ent_means, nmi_means, pc1_vals, wass_means];
metric_labels = {'PCA error (3c)', 'Temp entropy', 'All-pairs NMI', 'PC1%', 'Wasserstein'};
better_dirs   = {'lower', 'lower', 'higher', 'none', 'lower'};

% ── Figure 1: PCA error sorted boxplot ───────────────────────────────────────
draw_sorted_boxplot(pca_bmat, runs, ...
    'PCA reconstruction error (3 components) — lower = better  [EI mask per method]', ...
    'PCA recon error', 'lower', ...
    fullfile(out_dir, 'metric_pca_error.png'));

% ── Figure 2: Temporal entropy sorted boxplot ─────────────────────────────────
draw_sorted_boxplot(ent_bmat, runs, ...
    'Temporal entropy per voxel (8 bins, 16 tp) — lower = better  [EI mask per method]', ...
    'Temporal entropy (bits)', 'lower', ...
    fullfile(out_dir, 'metric_temporal_entropy.png'));

% ── Figure 3: All-pairs NMI sorted boxplot ───────────────────────────────────
draw_sorted_boxplot(nmi_vecs, runs, ...
    sprintf('All-pairs NMI  C(16,2)=%d pairs — higher = better', N_PAIRS), ...
    'NMI  [1,2]', 'higher', ...
    fullfile(out_dir, 'metric_allpairs_nmi.png'));

% ── Figure 4: Wasserstein distance sorted boxplot ─────────────────────────────
draw_sorted_boxplot(wass_vecs, runs, ...
    'Wasserstein dist: original vs registered per frame — lower = less data distortion', ...
    'Wasserstein distance (intensity units)', 'lower', ...
    fullfile(out_dir, 'metric_wasserstein.png'));

% ── Figure 5: PC1% bar chart ─────────────────────────────────────────────────
draw_pc1_bar(pc1_vals, runs, fullfile(out_dir, 'metric_pc1_bar.png'));

% ── Figure 5: Summary heatmap ─────────────────────────────────────────────────
draw_heatmap(metric_means, runs(:,1), metric_labels, better_dirs, ...
    fullfile(out_dir, 'metric_heatmap.png'));


% ══════════════════════════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS
% ══════════════════════════════════════════════════════════════════════════════

% ── Color lookup: letter code → RGB face color ────────────────────────────────
function fc = color_fc(c)
    switch c
        case 'b',  fc = [0.70 0.85 1.00];   % ANTs one-shot: blue
        case 'r',  fc = [1.00 0.80 0.80];   % Elx one-shot: red
        case 'g',  fc = [0.75 1.00 0.75];   % Full pipeline GW: green
        case 'm',  fc = [1.00 0.88 0.65];   % Full pipeline GW+ANTs: orange
        otherwise, fc = [0.82 0.82 0.82];   % original: grey
    end
end

% ── EI frame detection: frame with highest total signal sum ───────────────────
function ei = find_ei_4d(img4d)
    arr  = reshape(img4d, [], size(img4d,4));
    [~, ei] = max(sum(arr, 1));
end

% ── Wasserstein-1 distance between intensity histograms ──────────────────────
% Compares the intensity distribution of original_t vs registered_t within mask.
% W = integral |CDF_A - CDF_B| dx  (1D Wasserstein via CDF L1 norm)
% W = 0 → registration preserved signal exactly (just rearranged spatially)
% W > 0 → interpolation distorted signal values
function W = wasserstein_1d(a, b)
    a = double(a(:));
    b = double(b(:));
    n = 2048;   % quantile resolution
    q = linspace(0, 1, n);
    qa = quantile(a, q);
    qb = quantile(b, q);
    W  = mean(abs(qa - qb));
end

% ── NMI between two 3D volumes within mask ───────────────────────────────────
% Validation: A==B → NMI=2.0;  A⊥B (independent) → NMI=1.0
function nmi = compute_nmi(A, B, mask, nbins)
    a = double(A(mask(:)));
    b = double(B(mask(:)));
    a = (a - min(a)) / (max(a) - min(a) + eps);
    b = (b - min(b)) / (max(b) - min(b) + eps);
    edges = linspace(0, 1, nbins+1);
    H2  = histcounts2(a, b, edges, edges);
    H2  = H2 / (sum(H2(:)) + eps);
    pA  = sum(H2, 2);
    pB  = sum(H2, 1);
    H_A  = -sum(pA(pA>0) .* log2(pA(pA>0)));
    H_B  = -sum(pB(pB>0) .* log2(pB(pB>0)));
    H_AB = -sum(H2(H2>0) .* log2(H2(H2>0)));
    nmi  = (H_A + H_B) / (H_AB + eps);
end

% ── Temporal entropy per masked voxel (vectorised) ───────────────────────────
% Each voxel's 16-point time series → normalise → bin → Shannon entropy
function ent_vec = compute_temporal_entropy(img4d, mask, nbins)
    [X,Y,Z,T] = size(img4d);
    arr = reshape(img4d, X*Y*Z, T);
    arr = arr(mask(:), :);                          % (N_mask, T)
    rmin = min(arr, [], 2);
    rmax = max(arr, [], 2);
    arr_n = (arr - rmin) ./ (rmax - rmin + eps);    % normalise each row
    bidx  = min(floor(arr_n * nbins) + 1, nbins);   % bin index 1..nbins
    N     = size(arr_n, 1);
    ent_vec = zeros(N, 1);
    for b = 1:nbins
        p  = sum(bidx == b, 2) / T;
        nz = p > 0;
        ent_vec(nz) = ent_vec(nz) - p(nz) .* log2(p(nz));
    end
end

% ── PC1 explained variance % ─────────────────────────────────────────────────
function pc1 = compute_pc1(img4d, mask)
    [X,Y,Z,T] = size(img4d);
    arr = reshape(img4d, X*Y*Z, T);
    arr = arr(mask(:), :);
    [~,~,~,~,explained] = pca(arr);
    pc1 = explained(1);
end

% ── Sorted boxplot ────────────────────────────────────────────────────────────
function draw_sorted_boxplot(bmat, runs, title_str, ylabel_str, better_dir, out_path)
    N = size(bmat, 2);
    col_means = mean(bmat, 'omitnan');
    col_stds  = std(bmat,  'omitnan');
    if strcmp(better_dir, 'lower')
        [~, ord] = sort(col_means, 'ascend');
    else
        [~, ord] = sort(col_means, 'descend');
    end
    bmat_s   = bmat(:, ord);
    labels_s = runs(ord, 1);   % label is col 1
    colors_s = runs(ord, 2);   % color is col 2
    means_s  = col_means(ord);
    stds_s   = col_stds(ord);

    figure('Position', [50 50 2200 560]);
    bp = boxplot(bmat_s, 'Labels', labels_s, 'Notch','on','Symbol','','Whisker',1.5);
    ax = gca;
    ax.XTickLabelRotation = 35;
    ax.FontSize = 10;
    grid on;
    ylabel(ylabel_str);
    title(title_str, 'FontSize',11,'FontWeight','bold');

    av = bmat_s(~isnan(bmat_s));
    yl = [prctile(av,0.5) prctile(av,99.5)];
    ylim(yl);  yspan = yl(2)-yl(1);

    boxes = findobj(bp,'Tag','Box');
    for k = 1:numel(boxes)
        xd  = get(boxes(k),'XData');
        pos = round(mean(xd));          % actual column position 1..N
        pos = max(1, min(N, pos));
        c   = colors_s{pos};
        fc = color_fc(c);
        patch(xd, get(boxes(k),'YData'), fc, ...
              'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3]);
    end

    hold on;
    for k = 1:N
        text(k, yl(1)+0.10*yspan, sprintf('%.4f',means_s(k)), ...
             'HorizontalAlignment','center','FontSize',7,'Color',[0.1 0.1 0.55]);
        text(k, yl(1)+0.03*yspan, sprintf('±%.4f',stds_s(k)), ...
             'HorizontalAlignment','center','FontSize',7,'Color',[0.55 0.1 0.1]);
    end
    text(1,   yl(2)-0.03*yspan, '▲ best',  'HorizontalAlignment','center','FontSize',9,'Color',[0 0.5 0],'FontWeight','bold');
    text(N,   yl(2)-0.03*yspan, '▼ worst', 'HorizontalAlignment','center','FontSize',9,'Color',[0.7 0 0],'FontWeight','bold');

    patch(nan,nan,color_fc('b'),'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','ANTs one-shot');
    patch(nan,nan,color_fc('r'),'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','Elx one-shot');
    patch(nan,nan,color_fc('g'),'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','Full GW');
    patch(nan,nan,color_fc('m'),'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','Full GW+ANTs');
    patch(nan,nan,color_fc('k'),'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','original');
    legend('Location','northeast','FontSize',9);
    hold off;
    saveas(gcf, out_path);  fprintf('Saved: %s\n', out_path);
end

% ── PC1% bar chart ────────────────────────────────────────────────────────────
function draw_pc1_bar(pc1_vals, runs, out_path)
    N = numel(pc1_vals);
    figure('Position',[50 50 1400 450]);
    hold on;
    for k = 1:N
        c = runs{k,2};   % color is column 2
        bar(k, pc1_vals(k), 'FaceColor', color_fc(c), 'EdgeColor',[0.3 0.3 0.3]);
        text(k, pc1_vals(k)+0.3, sprintf('%.1f%%', pc1_vals(k)), ...
             'HorizontalAlignment','center','FontSize',7,'Color',[0.2 0.2 0.2]);
    end
    hold off;
    set(gca,'XTick',1:N,'XTickLabel',runs(:,1),'XTickLabelRotation',35,'FontSize',8);
    ylabel('PC1 explained variance %');
    title('PC1% — no directional interpretation for xenon MRI (shown for reference)', ...
          'FontSize',10,'FontWeight','bold');
    grid on;
    saveas(gcf, out_path);  fprintf('Saved: %s\n', out_path);
end

% ── Summary heatmap ───────────────────────────────────────────────────────────
function draw_heatmap(scalars, labels_run, labels_met, better_dirs, out_path)
    N_runs = size(scalars,1);
    N_met  = size(scalars,2);

    % Rank within each directional metric column (1=best), NaN rows excluded
    rank_mat = nan(N_runs, N_met);
    for m = 1:N_met
        if strcmp(better_dirs{m},'none'), continue; end
        v    = scalars(:,m);
        valid = find(~isnan(v));
        if isempty(valid), continue; end
        if strcmp(better_dirs{m},'lower')
            [~,sord] = sort(v(valid),'ascend');
        else
            [~,sord] = sort(v(valid),'descend');
        end
        rank_mat(valid(sord), m) = 1:numel(valid);
    end

    % Sort rows by mean rank of directional metrics
    dir_cols   = ~cellfun(@(x) strcmp(x,'none'), better_dirs);
    mean_rank  = mean(rank_mat(:,dir_cols), 2, 'omitnan');
    [~,row_ord] = sort(mean_rank,'ascend');

    rank_s    = rank_mat(row_ord,:);
    scalar_s  = scalars(row_ord,:);
    labels_s  = labels_run(row_ord);

    % Green → Red colormap  (rank 1 = green = best, rank N = red = worst)
    n_steps  = N_runs;
    t        = linspace(0,1,n_steps)';   % 0 = best (rank 1), 1 = worst (rank N)
    cmap_dir = [0.15 + t*0.75, ...       % R: low for green, high for red
                0.80 - t*0.70, ...       % G: high for green, low for red
                0.15*ones(n_steps,1)];   % B: constant low
    % For 'none' columns: grey scale by value
    cmap_grey = gray(n_steps);

    figure('Position',[50 50 1100 760]);
    ax = axes; hold on;
    ax.TickLength = [0 0];

    for r = 1:N_runs
        for m = 1:N_met
            val = scalar_s(r,m);
            if isnan(val)
                fc = [0.95 0.95 0.95];   % white = missing data
                txt = 'N/A';
            elseif strcmp(better_dirs{m},'none')
                col_vals = scalar_s(:,m);
                v_norm   = (val - min(col_vals(~isnan(col_vals)))) / ...
                           (max(col_vals(~isnan(col_vals))) - min(col_vals(~isnan(col_vals))) + eps);
                gv = 0.95 - v_norm*0.55;
                fc  = [gv gv gv];
                txt = sprintf('%.4f', val);
            else
                rk = rank_s(r,m);
                if isnan(rk), fc = [0.95 0.95 0.95]; txt = 'N/A';
                else, fc = cmap_dir(round(rk),:); txt = sprintf('%.4f', val); end
            end
            rectangle('Position',[m-0.5, r-0.5, 1, 1], ...
                      'FaceColor',fc,'EdgeColor','w','LineWidth',1.2);
            text(m, r, txt, 'HorizontalAlignment','center','FontSize',7, ...
                 'FontWeight','bold','Color',[0.1 0.1 0.1]);
        end
        if isnan(mean_rank(row_ord(r)))
            rank_txt = 'N/A';
        else
            rank_txt = sprintf('rank %.1f', mean_rank(row_ord(r)));
        end
        text(N_met+0.65, r, rank_txt, 'FontSize',8,'Color',[0.3 0.3 0.3]);
    end

    % Metric direction labels at top
    for m = 1:N_met
        if strcmp(better_dirs{m},'lower'),  d='(↓ better)';
        elseif strcmp(better_dirs{m},'higher'), d='(↑ better)';
        else, d='(ref only)'; end
        text(m, 0.1, d,'HorizontalAlignment','center','FontSize',7,'Color',[0.4 0.4 0.4]);
    end

    ax.XTick = 1:N_met;  ax.XTickLabel = labels_met;
    ax.XTickLabelRotation = 15;  ax.FontSize = 9;
    ax.YTick = 1:N_runs; ax.YTickLabel = labels_s;
    ax.YDir  = 'normal';
    xlim([0.5 N_met+1.2]);  ylim([0.5 N_runs+0.5]);
    title('Metric summary — rows sorted by mean rank across directional metrics', ...
          'FontSize',10,'FontWeight','bold');
    hold off;
    saveas(gcf, out_path);  fprintf('Saved: %s\n', out_path);
end
