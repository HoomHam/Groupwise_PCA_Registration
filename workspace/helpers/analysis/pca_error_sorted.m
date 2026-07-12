% pca_error_sorted.m
% Two metrics: PCA reconstruction error + mean temporal std.
% Four figures total: each metric sorted by mean, then by std.

input_dir      = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/';
comparison_dir = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/';
threshold = 0.04; n_components = 3; vote_thresh = 9;

%% Load original
rspace_files = dir(fullfile(input_dir, 'rspace*.mat'));
mat_raw = load(fullfile(input_dir, rspace_files(1).name), 'image');
original = double(mat_raw.image); original = original / max(original(:));
vm = sum(original >= threshold, 4) >= vote_thresh;
err_orig   = computePCAError(original, n_components) .* vm;
std_map_or = std(original, 0, 4);

%% All runs  {subdir, label, color}
runs = { ...
    'original',               'original',    'k'; ...
    'ants_image_only',        'ANTs img',    'b'; ...
    'ants_enhance_nonpad',    'ANTs enh-np', 'b'; ...
    'ants_enhance_pad',       'ANTs enh-p',  'b'; ...
    'ants_denoise_nonpad',    'ANTs den-np', 'b'; ...
    'ants_denoise_pad',       'ANTs den-p',  'b'; ...
    'ants_clahe_nonpad',      'ANTs clh-np', 'b'; ...
    'ants_clahe_pad',         'ANTs clh-p',  'b'; ...
    'elastix_image_only',     'Elx img',     'r'; ...
    'elastix_enhance_nonpad', 'Elx enh-np',  'r'; ...
    'elastix_enhance_pad',    'Elx enh-p',   'r'; ...
    'elastix_denoise_nonpad', 'Elx den-np',  'r'; ...
    'elastix_denoise_pad',    'Elx den-p',   'r'; ...
    'elastix_clahe_nonpad',   'Elx clh-np',  'r'; ...
    'elastix_clahe_pad',      'Elx clh-p',   'r'; ...
};
N = size(runs, 1);

%% Collect per-voxel vectors
pca_vecs = cell(N,1);
std_vecs = cell(N,1);

pca_vecs{1} = err_orig(vm);
std_vecs{1} = std_map_or(vm);

for r = 2:N
    p = fullfile(comparison_dir, runs{r,1}, 'gas.mat');
    s = load(p, 'gas');
    img = double(s.gas) / max(double(s.gas(:)));
    e = computePCAError(img, n_components) .* vm;
    pca_vecs{r} = e(vm);
    sm = std(img, 0, 4);
    std_vecs{r} = sm(vm);
end

%% Build full matrices
    function bmat = to_matrix(vecs)
        nr = max(cellfun(@numel, vecs));
        bmat = nan(nr, numel(vecs));
        for c = 1:numel(vecs)
            v = vecs{c}(:);
            bmat(1:numel(v), c) = v;
        end
    end

pca_bmat = to_matrix(pca_vecs);
std_bmat = to_matrix(std_vecs);

%% Draw one sorted figure
    function draw_sorted(bmat, sort_metric_bmat, metric_name, sort_by_stat, runs, out_path)
        N = size(bmat, 2);
        col_means = mean(bmat, 'omitnan');
        col_stds  = std(bmat,  'omitnan');

        % Sort by chosen stat on sort_metric_bmat
        sm_means = mean(sort_metric_bmat, 'omitnan');
        sm_stds  = std(sort_metric_bmat,  'omitnan');
        if strcmp(sort_by_stat, 'mean')
            [~, ord] = sort(sm_means, 'ascend');
            sort_desc = sprintf('sorted by mean %s (best = leftmost)', metric_name);
        else
            [~, ord] = sort(sm_stds, 'ascend');
            sort_desc = sprintf('sorted by std %s (most consistent = leftmost)', metric_name);
        end

        bmat_s    = bmat(:, ord);
        labels_s  = runs(ord, 2);
        colors_s  = runs(ord, 3);
        means_s   = col_means(ord);
        stds_s    = col_stds(ord);

        figure('Name', sort_desc, 'Position', [50 50 1500 520]);
        ax = axes;
        bp = boxplot(bmat_s, 'Labels', labels_s, 'Notch','on', ...
                     'Symbol','', 'Whisker',1.5);
        ax.XTickLabelRotation = 35;
        ax.FontSize = 10;
        grid on;
        ylabel(metric_name);
        title(sort_desc, 'FontSize', 11, 'FontWeight', 'bold');

        all_vals = bmat_s(~isnan(bmat_s));
        yl = [prctile(all_vals,0.5) prctile(all_vals,99.5)];
        ylim(yl);

        % Colour boxes
        boxes = findobj(bp, 'Tag', 'Box');
        for k = 1:numel(boxes)
            col_idx = N - k + 1;
            c = colors_s{col_idx};
            if strcmp(c,'b'),     fc = [0.70 0.85 1.00];
            elseif strcmp(c,'r'), fc = [1.00 0.80 0.80];
            else,                 fc = [0.82 0.82 0.82];
            end
            patch(get(boxes(k),'XData'), get(boxes(k),'YData'), fc, ...
                  'FaceAlpha',0.65, 'EdgeColor',[0.3 0.3 0.3]);
        end

        % Annotate mean ± std below each box
        hold on;
        y_mean = yl(1) + 0.10*(yl(2)-yl(1));
        y_std  = yl(1) + 0.03*(yl(2)-yl(1));
        for k = 1:N
            text(k, y_mean, sprintf('%.4f', means_s(k)), ...
                 'HorizontalAlignment','center','FontSize',7,'Color',[0.15 0.15 0.5]);
            text(k, y_std, sprintf('±%.4f', stds_s(k)), ...
                 'HorizontalAlignment','center','FontSize',7,'Color',[0.5 0.15 0.15]);
        end

        % Legend
        patch(nan,nan,[0.70 0.85 1.00],'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','ANTs');
        patch(nan,nan,[1.00 0.80 0.80],'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','Elastix');
        patch(nan,nan,[0.82 0.82 0.82],'FaceAlpha',0.65,'EdgeColor',[0.3 0.3 0.3],'DisplayName','original');
        legend('Location','northeast','FontSize',9);
        hold off;

        saveas(gcf, out_path);
        fprintf('Saved: %s\n', out_path);
    end

%% PCA error — sorted by mean and by std
draw_sorted(pca_bmat, pca_bmat, 'PCA recon error (3 comp)', 'mean', runs, ...
    fullfile(comparison_dir, 'sorted_pca_by_mean.png'));

draw_sorted(pca_bmat, pca_bmat, 'PCA recon error (3 comp)', 'std', runs, ...
    fullfile(comparison_dir, 'sorted_pca_by_std.png'));

%% Temporal std — sorted by mean and by std
draw_sorted(std_bmat, std_bmat, 'Mean temporal std', 'mean', runs, ...
    fullfile(comparison_dir, 'sorted_tempstd_by_mean.png'));

draw_sorted(std_bmat, std_bmat, 'Mean temporal std', 'std', runs, ...
    fullfile(comparison_dir, 'sorted_tempstd_by_std.png'));
