% pca_error_comparison.m
% Separate ANTs vs Elastix panels.
% Annotates each method with % improvement over its own image-only baseline.

input_dir      = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/';
comparison_dir = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/';

threshold    = 0.04;
n_components = 3;
vote_thresh  = 9;

%% Load original
rspace_files = dir(fullfile(input_dir, 'rspace*.mat'));
mat_raw = load(fullfile(input_dir, rspace_files(1).name), 'image');
original = double(mat_raw.image);
original = original / max(original(:));
vm = sum(original >= threshold, 4) >= vote_thresh;
err_original = computePCAError(original, n_components);
err_original = err_original .* vm;

%% Define runs per method
% {subdir, short_label, group}  group: 1=img, 2=enhance, 3=denoise, 4=clahe
ants_runs = { ...
    'ants_image_only',       'img',       1; ...
    'ants_enhance_nonpad',   'enh-np',    2; ...
    'ants_enhance_pad',      'enh-p',     2; ...
    'ants_denoise_nonpad',   'den-np',    3; ...
    'ants_denoise_pad',      'den-p',     3; ...
    'ants_clahe_nonpad',     'clahe-np',  4; ...
    'ants_clahe_pad',        'clahe-p',   4; ...
};

elx_runs = { ...
    'elastix_image_only',       'img',       1; ...
    'elastix_enhance_nonpad',   'enh-np',    2; ...
    'elastix_enhance_pad',      'enh-p',     2; ...
    'elastix_denoise_nonpad',   'den-np',    3; ...
    'elastix_denoise_pad',      'den-p',     3; ...
    'elastix_clahe_nonpad',     'clahe-np',  4; ...
    'elastix_clahe_pad',        'clahe-p',   4; ...
};

%% Load errors helper
    function errs = load_errors(run_list, comp_dir, mask, n_comp)
        errs = cell(size(run_list,1), 1);
        for r = 1:size(run_list,1)
            p = fullfile(comp_dir, run_list{r,1}, 'gas.mat');
            if ~isfile(p); errs{r} = []; continue; end
            s   = load(p, 'gas');
            img = double(s.gas) / max(double(s.gas(:)));
            e   = computePCAError(img, n_comp);
            errs{r} = (e .* mask);
        end
    end

ants_errs = load_errors(ants_runs, comparison_dir, vm, n_components);
elx_errs  = load_errors(elx_runs,  comparison_dir, vm, n_components);

%% Build boxplot matrix + compute % improvement vs image-only baseline
    function [bmat, labels, pct_improv] = build_matrix(run_list, errs, err_orig, mask)
        % prepend original
        all_data   = [{err_orig(mask)}, cellfun(@(e) e(mask), errs, 'UniformOutput', false)'];
        all_labels = [{'original'}, run_list(:,2)'];
        n_cols = numel(all_data);
        n_rows = max(cellfun(@numel, all_data));
        bmat   = nan(n_rows, n_cols);
        for c = 1:n_cols
            if ~isempty(all_data{c})
                v = all_data{c}(:);
                bmat(1:numel(v), c) = v;
            end
        end
        labels = all_labels;

        % % improvement vs image-only (column 2 = first registered run)
        med_img_only = median(bmat(:,2), 'omitnan');
        pct_improv   = nan(1, n_cols);
        for c = 2:n_cols
            med_c = median(bmat(:,c), 'omitnan');
            pct_improv(c) = (med_img_only - med_c) / med_img_only * 100;
        end
        % original vs img-only too
        med_orig = median(bmat(:,1), 'omitnan');
        pct_improv(1) = (med_img_only - med_orig) / med_img_only * 100;
    end

[ants_bmat, ants_labels, ants_pct] = build_matrix(ants_runs, ants_errs, err_original, vm);
[elx_bmat,  elx_labels,  elx_pct]  = build_matrix(elx_runs,  elx_errs,  err_original, vm);

%% Draw panels
    function draw_panel(bmat, labels, pct, run_list, title_str)
        boxplot(bmat, 'Labels', labels, 'Notch','on', 'Symbol','', 'Whisker',1.5);
        group_colors = [0.75 0.75 0.75;   % grey  — image-only
                        0.70 0.85 1.00;   % blue  — enhance
                        0.75 1.00 0.80;   % green — denoise
                        1.00 0.85 0.70];  % orange— clahe
        ax = gca;
        ax.XTickLabelRotation = 30;
        ax.FontSize = 10;
        ylabel('PCA recon error (3 comp)');
        title(title_str, 'FontSize', 12, 'FontWeight','bold');
        grid on;

        all_vals = bmat(~isnan(bmat));
        yl = [prctile(all_vals,1) prctile(all_vals,99)];
        ylim(yl);
        hold on;

        % shade groups (col 1 = original, col 2+ = runs)
        group_ids = [0; cell2mat(run_list(:,3))];  % 0 for original
        for g = 1:4
            cols = find(group_ids == g);
            if isempty(cols); continue; end
            x1 = cols(1) - 0.5; x2 = cols(end) + 0.5;
            patch(ax, [x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], ...
                  group_colors(g,:), 'FaceAlpha',0.3, 'EdgeColor','none');
        end

        % annotate % improvement above each box
        y_ann = yl(1) + 0.94*(yl(2)-yl(1));
        for c = 1:numel(pct)
            if isnan(pct(c)); continue; end
            if c == 1
                txt = sprintf('%.1f%%', pct(c));   % original vs baseline
                col = [0.4 0.4 0.4];
            elseif pct(c) >= 0
                txt = sprintf('+%.1f%%', pct(c));
                col = [0.0 0.55 0.1];
            else
                txt = sprintf('%.1f%%', pct(c));
                col = [0.75 0.1 0.1];
            end
            text(c, y_ann, txt, 'HorizontalAlignment','center', ...
                 'FontSize',8, 'FontWeight','bold', 'Color', col);
        end
        hold off;
    end

%% Plot
figure('Name','PCA error: ANTs vs Elastix','Position',[50 50 1400 700]);

subplot(2,1,1);
draw_panel(ants_bmat, ants_labels, ants_pct, ants_runs, ...
    sprintf('ANTs SyNCC — improvement vs ANTs image-only (median, %d PCA components)', n_components));

subplot(2,1,2);
draw_panel(elx_bmat, elx_labels, elx_pct, elx_runs, ...
    sprintf('Elastix groupwise PCA — improvement vs Elastix image-only (median, %d PCA components)', n_components));

out_png = fullfile(comparison_dir, 'pca_error_comparison.png');
saveas(gcf, out_png);
fprintf('Saved: %s\n', out_png);
