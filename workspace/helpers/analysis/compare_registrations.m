% compare_registrations.m
% Compare registration quality across 4 iteration schedules × 2 final stages.
% Metrics: (1) mean temporal std within mask, (2) PCA explained variance.

base = '/Volumes/HoomHamExt/Work/Analysis/2024-11-13_025JC/reg/';

runs = { ...
    'test_543200iter',   '[500,400,300,200]'; ...
    'test_1000iter',     '[1000]'; ...
    'test_5432000iter_nofilter',  '[5000,4000,3000,2000]'; ...
    'test_5432000iter',  '[5000,4000,3000,2000]'; ...
    'test_5000iter',     '[5000]'; ...
};
n_runs = size(runs, 1);

stages   = {'final_groupwise', 'final_ants'};
n_stages = numel(stages);

channels = {'gas', 'rbc', 'mem'};
n_ch     = numel(channels);

% mean_std / pc1_var / pc2_var: [n_runs x n_ch x n_stages]
mean_std   = nan(n_runs, n_ch, n_stages);
pc1_var    = nan(n_runs, n_ch, n_stages);
pc2_var    = nan(n_runs, n_ch, n_stages);
std_vox      = cell(n_runs, n_stages);   % full voxel-wise std distributions (gas only)
pc_explained = cell(n_runs, n_stages);  % full explained variance spectrum (gas only)
pca_err_vox  = cell(n_runs, n_stages);  % per-voxel PCA reconstruction error (gas only)

for s = 1:n_stages
    for r = 1:n_runs
        fdir      = fullfile(base, runs{r,1}, stages{s});
        mask_dir  = fullfile(base, runs{r,1}, 'final_groupwise');
        mask4d    = double(niftiread(fullfile(mask_dir, 'mask.nii')));
        mask      = mask4d(:,:,:,1) > 0.5;

        nii_path = fullfile(fdir, [channels{1} '.nii']);
        if ~isfile(nii_path)
            fprintf('SKIP (missing): %s  /  %s\n', stages{s}, runs{r,2});
            continue;
        end

        for c = 1:n_ch
            vol = double(abs(niftiread(fullfile(fdir, [channels{c} '.nii']))));
            [X, Y, Z, T] = size(vol);

            std_map             = std(vol, 0, 4);
            mean_std(r, c, s)   = mean(std_map(mask));
            if c == 1
                std_vox{r, s}   = std_map(mask);   % gas only, full distribution
            end

            flat  = reshape(vol, X*Y*Z, T)';
            data  = flat(:, mask(:));
            [~, ~, ~, ~, explained] = pca(data);
            pc1_var(r, c, s) = explained(1);
            pc2_var(r, c, s) = explained(2);
            if c == 1
                pc_explained{r, s} = explained(1:6);   % top 6 PCs (rest near zero)
                err_map            = computePCAError(vol, 5);
                pca_err_vox{r, s}  = err_map(mask);
            end
        end
        fprintf('Done: %s  /  %s\n', stages{s}, runs{r,2});
    end
end

%% Print table — one block per stage, gas only
fprintf('\n=== Gas channel ===\n');
for s = 1:n_stages
    % ranks within this stage
    [~, idx] = sort(mean_std(:, 1, s), 'ascend');
    std_rank = zeros(n_runs, 1);
    std_rank(idx) = 1:n_runs;

    fprintf('\n[%s]\n', stages{s});
    fprintf('%-26s  %16s  %6s  %10s  %10s\n', ...
            'Iterations', 'Mean Temporal Std', 'Rank', 'PC1 Var%', 'PC2 Var%');
    fprintf('%s\n', repmat('-', 1, 76));
    for r = 1:n_runs
        fprintf('%-26s  %16.5f  %6d  %10.2f  %10.2f\n', ...
            runs{r,2}, mean_std(r,1,s), std_rank(r), pc1_var(r,1,s), pc2_var(r,1,s));
    end
end

%% Print groupwise vs ANTs delta for gas
fprintf('\n[groupwise → ANTs delta (gas)]\n');
fprintf('%-26s  %16s  %10s\n', 'Iterations', 'Δ Std (gw→ants)', 'Δ PC1% (gw→ants)');
fprintf('%s\n', repmat('-', 1, 56));
for r = 1:n_runs
    d_std = mean_std(r,1,2) - mean_std(r,1,1);
    d_pc1 = pc1_var(r,1,2)  - pc1_var(r,1,1);
    fprintf('%-26s  %+16.5f  %+10.2f\n', runs{r,2}, d_std, d_pc1);
end

%% Figure — gas only, both stages side by side
run_labels = runs(:,2);
gw_col     = [0.2 0.5 0.8];
an_col     = [0.9 0.4 0.1];

figure('Name', 'Gas: groupwise vs ANTs', 'Position', [100 100 1600 550]);

% --- Subplot 1: temporal std boxplots ---
subplot(1,2,1);

% Interleave gw/ants columns: gw_run1, ants_run1, gw_run2, ants_run2, ...
cols       = {};
box_labels = {};
box_colors = [];
for r = 1:n_runs
    if ~isempty(std_vox{r,1})
        cols{end+1}        = std_vox{r,1}(:);
        box_labels{end+1}  = sprintf('gw %s', run_labels{r});
        box_colors         = [box_colors; gw_col];
    end
    if ~isempty(std_vox{r,2})
        cols{end+1}        = std_vox{r,2}(:);
        box_labels{end+1}  = sprintf('ants %s', run_labels{r});
        box_colors         = [box_colors; an_col];
    end
end
% Pad columns with NaN so they're equal length
n_rows   = max(cellfun(@numel, cols));
box_data = nan(n_rows, numel(cols));
for k = 1:numel(cols)
    box_data(1:numel(cols{k}), k) = cols{k};
end

h = boxplot(box_data, 'Labels', box_labels, 'Symbol', '', 'Whisker', 1.5, 'Notch', 'on');
set(h, 'LineWidth', 1.2);

% Color boxes by stage
boxes = findobj(gca, 'Tag', 'Box');
boxes = flipud(boxes);   % boxplot returns in reverse order
for k = 1:numel(boxes)
    patch(get(boxes(k), 'XData'), get(boxes(k), 'YData'), box_colors(k,:), ...
          'FaceAlpha', 0.5, 'EdgeColor', box_colors(k,:));
end

set(gca, 'XTickLabelRotation', 25);
ylabel('Voxel-wise temporal std');
title('Gas — temporal std (lower=better)');
grid on;

% Manual legend
hold on;
p1 = patch(nan, nan, gw_col, 'FaceAlpha', 0.5, 'EdgeColor', gw_col);
p2 = patch(nan, nan, an_col, 'FaceAlpha', 0.5, 'EdgeColor', an_col);
legend([p1 p2], {'groupwise', 'ANTs'}, 'Location', 'northeast');
hold off;
ylim([0.0 0.07])

% --- Subplot 2: PC1 explained variance bar chart ---
% subplot(1,3,2);
% x  = 1:n_runs;
% bw = 0.35;
% hold on;
% bar(x - bw/2, pc1_var(:,1,1), bw, 'FaceColor', gw_col, 'DisplayName', 'groupwise');
% bar(x + bw/2, pc1_var(:,1,2), bw, 'FaceColor', an_col, 'DisplayName', 'ANTs');
% set(gca, 'XTick', x, 'XTickLabel', run_labels, 'XTickLabelRotation', 20);
% ylabel('PC1 explained var %');
% title('Gas — PC1 var% (higher=better)');
% legend('Location', 'southeast');
% grid on;
% hold off;


% --- Subplot 3: PCA reconstruction error boxplot ---
subplot(1,2,2);

err_cols   = {};
err_labels = {};
err_colors = [];
for r = 1:n_runs
    if ~isempty(pca_err_vox{r,1})
        err_cols{end+1}   = pca_err_vox{r,1}(:);
        err_labels{end+1} = sprintf('gw %s', run_labels{r});
        err_colors        = [err_colors; gw_col];
    end
    if ~isempty(pca_err_vox{r,2})
        err_cols{end+1}   = pca_err_vox{r,2}(:);
        err_labels{end+1} = sprintf('ants %s', run_labels{r});
        err_colors        = [err_colors; an_col];
    end
end
n_err_rows = max(cellfun(@numel, err_cols));
err_data   = nan(n_err_rows, numel(err_cols));
for k = 1:numel(err_cols)
    err_data(1:numel(err_cols{k}), k) = err_cols{k};
end

h3 = boxplot(err_data, 'Labels', err_labels, 'Symbol', '', 'Whisker', 1.5);
set(h3, 'LineWidth', 1.2);

boxes3 = findobj(gca, 'Tag', 'Box');
boxes3 = flipud(boxes3);
for k = 1:numel(boxes3)
    patch(get(boxes3(k), 'XData'), get(boxes3(k), 'YData'), err_colors(k,:), ...
          'FaceAlpha', 0.5, 'EdgeColor', err_colors(k,:));
end

set(gca, 'XTickLabelRotation', 25);
ylabel('PCA reconstruction error (3 components)');
title('Gas — PCA recon error (lower=better)');
grid on;

hold on;
p1 = patch(nan, nan, gw_col, 'FaceAlpha', 0.5, 'EdgeColor', gw_col);
p2 = patch(nan, nan, an_col, 'FaceAlpha', 0.5, 'EdgeColor', an_col);
legend([p1 p2], {'groupwise', 'ANTs'}, 'Location', 'northeast');
hold off;

sgtitle('Registration quality: groupwise vs ANTs — gas channel');
saveas(gcf, fullfile(base, 'registration_comparison_gas.png'));
fprintf('\nFigure saved to %s\n', fullfile(base, 'registration_comparison_gas.png'));

ylim([0.002 0.005])