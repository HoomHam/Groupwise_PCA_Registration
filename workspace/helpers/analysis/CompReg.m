function CompReg(img, imlabels, thresholds, visualizeFlag)

% 'sd' : squared differences
% 'mi' : normalized mutual information
% 'ld' : log absolute difference
% 'cc' : cross correlation

% =========== PARAMETERS ===========
figure(200); clf;

% =========== 1) LOAD & CREATE THRESHOLD MASKS ===========
images = struct();
masks  = struct();

for idx = 1:size(img,5),

    Il = squeeze(img(:,:,:,:,idx));

    images.(imlabels{idx}) = Il;  % no resizing or flipping
    threshold = thresholds(idx);

    % Normalize
    images.(imlabels{idx}) = images.(imlabels{idx}) / max(images.(imlabels{idx})(:));

    % Threshold-based mask
    local_mask = create_mask(images.(imlabels{idx}), threshold);
    masks.(imlabels{idx}) = local_mask;
end

% =========== 2) COMPUTE METRICS (UPPER TRIANGLE) ===========
metrics = compute_metrics(images, masks, imlabels);

% =========== 3) (Optional) VISUALIZE SLICES ===========
if visualizeFlag
    visualize_slices(images, masks, imlabels);
end

% =========== 4) MIN/MAX FOR MEANS & STDS ===========
metric_limits = compute_metric_limits(metrics);
std_limits    = compute_std_limits(metrics);

% =========== 5) PERCENTAGE DIFFS ===========
percentage_diffs = compute_percentage_diffs(metrics);

% =========== 6) PLOT (4x9) ===========
plot_metrics_means_only(metrics, metric_limits, std_limits, percentage_diffs, imlabels);

clc;
end

%% =============== FUNCTIONS ===============

function mask = create_mask(image, threshold)
mask = image;
mask(mask < threshold) = 0;
mask(mask > 0) = 1;
end

function visualize_slices(images,masks, imlabels)
figure(111);
method_for_mask = imlabels{1};
frames_count = size(images.(imlabels{1}),4);

for i = 1:frames_count
    clf;
    for slice_idx = 1:size(images.(imlabels{1}),2)
        for idx = 1:numel(imlabels)
            image_name = imlabels{idx};
            img_slice  = squeeze(images.(image_name)(:, slice_idx, :, i));
            mask_slice = squeeze(masks.(method_for_mask)(:, slice_idx, :, i));

            subplot(2,2,2*(idx-1)+1);
            imagesc(img_slice); colormap gray; axis image; colorbar;
            title([image_name ' Image (Frame ' num2str(i) ')']);

            subplot(2,2,2*(idx-1)+2);
            imagesc(mask_slice); colormap jet; axis image; colorbar;
            title([image_name ' Mask (Frame ' num2str(i) ')']);
        end
        pause(0.1);
    end
end
end

function metrics = compute_metrics(images, masks, imlabels)
% skip STD for 'mi' & 'cc', do upper triangle only
metric_types = {'sd','mi','ld','cc'};
methods      = imlabels;
metrics      = struct();

frames_count = size(images.(methods{1}),4);

for i = 1:frames_count
    % Normalize i-th frame
    for m = 1:numel(methods)
        method = methods{m};
        Imasked = squeeze(images.(method)(:,:,:,i));
        Imasked(masks.(method)(:,:,:,i)==0) = NaN;
        In = normalize_image(Imasked);
        images_n.(method).normalized{i} = In;
    end

    % Lower triangle => NaN
    for k = 1:i
        for m = 1:numel(methods)
            method = methods{m};
            for mt_idx = 1:numel(metric_types)
                metric_type = metric_types{mt_idx};
                metrics.(metric_type).mean.(method)(i,k) = NaN;
                if ~(strcmp(metric_type,'mi') || strcmp(metric_type,'cc'))
                    metrics.(metric_type).std.(method)(i,k) = NaN;
                end
            end
        end
    end

    for j = i:frames_count
        % Normalize j-th frame
        for m = 1:numel(methods)
            method = methods{m};
            Jmasked = squeeze(images.(method)(:,:,:,j));
            Jmasked(masks.(method)(:,:,:,j)==0) = NaN;
            Jn = normalize_image(Jmasked);
            images_n.(method).target_normalized{j} = Jn;
        end

        for mt_idx = 1:numel(metric_types)
            metric_type = metric_types{mt_idx};
            for m = 1:numel(methods)
                method = methods{m};
                [val,Irn] = image_difference_HH(images_n.(method).normalized{i},...
                    images_n.(method).target_normalized{j},metric_type,masks.(method)(:,:,:,j));
                metrics.(metric_type).mean.(method)(i,j)=val;

                if sum(Irn,[1,2,3])~=0
                    Irn(Irn==0)=NaN;
                end
                if ~(strcmp(metric_type,'mi')||strcmp(metric_type,'cc'))
                    metrics.(metric_type).std.(method)(i,j)=nanstd(Irn(:));
                end
            end
        end
    end
end
end

function normalized_image = normalize_image(image)
normalized_image = image / nanmean(image(:));
normalized_image(isnan(normalized_image))=0;
end

function metric_limits = compute_metric_limits(metrics)

metric_types = fieldnames(metrics);
metric_limits = struct();
for i=1:numel(metric_types)
    type = metric_types{i};
    method_fields = fieldnames(metrics.(type).mean);

    combined=[];
    for m=1:numel(method_fields)
        method=method_fields{m};
        combined=[combined; metrics.(type).mean.(method)(:)];
    end
    metric_limits.(type).min = nanmin(combined);
    metric_limits.(type).max = nanmax(combined);
end
end

function std_limits = compute_std_limits(metrics)
metric_types = fieldnames(metrics);
std_limits = struct();
for i=1:numel(metric_types)
    type = metric_types{i};
    if ~isfield(metrics.(type),'std'), continue; end

    method_fields=fieldnames(metrics.(type).std);
    combined=[];
    for m=1:numel(method_fields)
        method=method_fields{m};
        combined=[combined; metrics.(type).std.(method)(:)];
    end
    std_limits.(type).min = nanmin(combined);
    std_limits.(type).max = nanmax(combined);
end
end

function percentage_diffs = compute_percentage_diffs(metrics)
metric_types = fieldnames(metrics);
percentage_diffs=struct();
for i=1:numel(metric_types)
    metric = metric_types{i};
    methods_in_type=fieldnames(metrics.(metric).mean);
    if numel(methods_in_type)<2
        percentage_diffs.(metric).diffPerc=NaN;
        continue;
    end
    methodA=methods_in_type{1};
    methodB=methods_in_type{2};

    valsA=metrics.(metric).mean.(methodA)(:);
    valsB=metrics.(metric).mean.(methodB)(:);

    percentage_diffs.(metric).diffPerc=(nansum(valsB)-nansum(valsA)) / nansum(valsA)*100;
end
end

function plot_metrics(metrics, metric_limits, std_limits, percentage_diffs, imlabels)
load('colormapdiff.mat')

% columns in 4x9:
% 1=ANT Mean,2=Elx Mean,3=Mean Diff,4=Mean Boxplot,5=empty,
% 6=ANT STD,7=Elx STD,8=STD Diff,9=STD Boxplot
% skip col6..9 for mi,cc
plot_order = {'sd','mi','ld','cc'};
methods=imlabels;
methodA=methods{1}; methodB=methods{2};

subplot_indices_ANT_Mean  =[1, 10, 19, 28];
subplot_indices_Elx_Mean  =[2, 11, 20, 29];
subplot_indices_Diff_Mean =[3, 12, 21, 30];
subplot_indices_Box_Mean  =[4, 13, 22, 31];
subplot_indices_Empty     =[5, 14, 23, 32];
subplot_indices_ANT_STD   =[6, 15, 24, 33];
subplot_indices_Elx_STD   =[7, 16, 25, 34];
subplot_indices_Diff_STD  =[8, 17, 26, 35];
subplot_indices_Box_STD   =[9, 18, 27, 36];

for idx=1:numel(plot_order)
    type=plot_order{idx};

    antM_idx = subplot_indices_ANT_Mean(idx);
    elxM_idx = subplot_indices_Elx_Mean(idx);
    difM_idx = subplot_indices_Diff_Mean(idx);
    boxM_idx = subplot_indices_Box_Mean(idx);
    emp_idx  = subplot_indices_Empty(idx);
    antS_idx = subplot_indices_ANT_STD(idx);
    elxS_idx = subplot_indices_Elx_STD(idx);
    difS_idx = subplot_indices_Diff_STD(idx);
    boxS_idx = subplot_indices_Box_STD(idx);

    % ============== MEAN (methodA)
    ax1=subplot(4,9,antM_idx);
    matA=metrics.(type).mean.(methodA);
    imagesc(matA,'AlphaData',~isnan(matA));
    colormap(ax1,parula); set(gca,'Color','white');
    title([methodA ' ' type ' Mean']); axis image; colorbar;
    caxis([metric_limits.(type).min metric_limits.(type).max]);

    % ============== MEAN (methodB)
    ax1=subplot(4,9,elxM_idx);
    matB=metrics.(type).mean.(methodB);
    imagesc(matB,'AlphaData',~isnan(matB));
    colormap(ax1,parula); set(gca,'Color','white');
    title([methodB ' ' type ' Mean']); axis image; colorbar;
    caxis([metric_limits.(type).min metric_limits.(type).max]);

    % ============== MEAN Diff
    ax2=subplot(4,9,difM_idx);
    if plot_order{idx} == 'mi'
        diffM=(matA - matB)./matA*100;
    else
        diffM=(matB - matA)./matA*100;
    end
    ylabel('[%]')
    imagesc(diffM,'AlphaData',~isnan(diffM));
    colormap(ax2,cdiff); set(gca,'Color','white');
    title([type ' Mean Diff']); axis image; colorbar;
    dLim=max(abs(diffM(:)),[],'omitnan');
    caxis([-dLim dLim]);

    % ============== MEAN Boxplot
    subplot(4,9,boxM_idx);
    boxplot([matA(:), matB(:)], 'Labels',{methodA,methodB});
    title([type ' Mean Boxplot']);
    set(gca,'XTickLabelRotation',45);
    gird on

    % ============== Empty column
    subplot(4,9,emp_idx); cla; axis off;

    skipSTD=(strcmp(type,'mi')||strcmp(type,'cc'));
    if skipSTD
        % Clear out col6..9
        subplot(4,9,antS_idx); cla; axis off;
        subplot(4,9,elxS_idx); cla; axis off;
        subplot(4,9,difS_idx); cla; axis off;
        subplot(4,9,boxS_idx); cla; axis off;
        continue;
    end

    % ============== STD (methodA)
    ax1=subplot(4,9,antS_idx);
    stdA=metrics.(type).std.(methodA);
    imagesc(stdA,'AlphaData',~isnan(stdA));
    colormap(ax1,parula); set(gca,'Color','white');
    title([methodA ' ' type ' STD']); axis image; colorbar;
    caxis([std_limits.(type).min std_limits.(type).max]);

    % ============== STD (methodB)
    ax1=subplot(4,9,elxS_idx);
    stdB=metrics.(type).std.(methodB);
    imagesc(stdB,'AlphaData',~isnan(stdB));
    colormap(ax1,parula); set(gca,'Color','white');
    title([methodB ' ' type ' STD']); axis image; colorbar;
    caxis([std_limits.(type).min std_limits.(type).max]);

    % ============== STD Diff
    ax2=subplot(4,9,difS_idx);
    diffS=(stdB - stdA)./stdA*100;
    ylabel('[%]')
    imagesc(diffS,'AlphaData',~isnan(diffS));
    colormap(ax2,cdiff); set(gca,'Color','white');
    title([type ' STD Diff']); axis image; colorbar;
    dsLim=max(abs(diffS(:)),[],'omitnan');
    caxis([-dsLim dsLim]);

    % ============== STD Boxplot
    subplot(4,9,boxS_idx);
    boxplot([stdA(:), stdB(:)], 'Labels',{methodA,methodB});
    title([type ' STD Boxplot']);
    set(gca,'XTickLabelRotation',45);
end

% 4) Add annotation text for percentage differences
text_x       = 0.65;
text_y_start = 0.14;
text_spacing = 0.03;
plotOrder=fieldnames(percentage_diffs);

for i=1:numel(plotOrder)
    t=plotOrder{i};
    diffp=percentage_diffs.(t).diffPerc;
    if strcmp(t,'mi')
        is_better = diffp>0;
    else
        is_better = diffp<0;
    end
    annotation_text=sprintf('%s was %.2f%% %s in %s compared to %s',...
        methodB, abs(diffp), ternary(is_better,'better','worse'), upper(t), methodA);
    annotation('textbox',[text_x, text_y_start-(i-1)*text_spacing,0.4,0.05],...
        'String',annotation_text,'FitBoxToText','on','EdgeColor','none',...
        'FontSize',10,'FontWeight','bold');
end
end

function plot_metrics_means_only(metrics, metric_limits, std_limits, percentage_diffs, imlabels)
    load('colormapdiff.mat');

    % Columns: 1=ANT Mean, 2=Elx Mean, 3=Mean % Diff, 4=Boxplot, 5=Spacer
    plot_order = {'sd','mi','ld','cc'};
    methods = imlabels;
    methodA = methods{1};
    methodB = methods{2};

    subplot_indices_ANT_Mean  = [1, 5, 9, 13];
    subplot_indices_Elx_Mean  = [2, 6, 10, 14];
    subplot_indices_Diff_Mean = [3, 7, 11, 15];
    subplot_indices_Box_Mean  = [4, 8, 12, 16];

    for idx = 1:numel(plot_order)
        type = plot_order{idx};

        antM_idx = subplot_indices_ANT_Mean(idx);
        elxM_idx = subplot_indices_Elx_Mean(idx);
        difM_idx = subplot_indices_Diff_Mean(idx);
        boxM_idx = subplot_indices_Box_Mean(idx);
        % emp_idx  = subplot_indices_Empty(idx);

        % === MEAN (methodA) ===
        ax1 = subplot(4, 4, antM_idx);
        matA = metrics.(type).mean.(methodA);
        imagesc(matA, 'AlphaData', ~isnan(matA));
        colormap(ax1, parula); set(gca, 'Color', 'white');
        title([methodA ' ' upper(type) ' Mean']); axis image; colorbar;
        caxis([metric_limits.(type).min metric_limits.(type).max]);

        % === MEAN (methodB) ===
        ax2 = subplot(4, 4, elxM_idx);
        matB = metrics.(type).mean.(methodB);
        imagesc(matB, 'AlphaData', ~isnan(matB));
        colormap(ax2, parula); set(gca, 'Color', 'white');
        title([methodB ' ' upper(type) ' Mean']); axis image; colorbar;
        caxis([metric_limits.(type).min metric_limits.(type).max]);

        % === MEAN % DIFF ===
        ax3 = subplot(4, 4, difM_idx);
        % diffM = (matB - matA) ./ matA * 100;
        if plot_order{idx} == 'mi'
            diffM=(matA - matB)./matA*100;
        else
            diffM=(matB - matA)./matA*100;
        end
        imagesc(diffM, 'AlphaData', ~isnan(diffM));
        colormap(ax3, cdiff); set(gca, 'Color', 'white');
        title([upper(type) ' Mean Diff (%)']); axis image; colorbar;
        dLim = max(abs(diffM(:)), [], 'omitnan');
        caxis([-dLim dLim]);
        % caxis([-50 50]);

        % === BOX PLOT ===
        subplot(4, 4, boxM_idx);
        boxplot([matA(:), matB(:)], 'Labels', {methodA, methodB});
        title([upper(type) ' Mean Boxplot']);
        set(gca, 'XTickLabelRotation', 45);
        grid on

        % === EMPTY COLUMN ===
        % subplot(4, 4, emp_idx); cla; axis off;
    end
end




function result = ternary(condition, tv, fv)
if condition, result=tv; else, result=fv; end
end