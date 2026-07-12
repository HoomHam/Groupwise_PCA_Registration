input_dir      = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/rec/';
comparison_dir = '/Users/hoomham/Hooman/Work/Analysis/2024-11-13_025JC/reg/comparison/';
THRESHOLD=0.04; N_COMP=3; VOTE_THRESH=9; NMI_BINS=64; ENT_BINS=8;

labels  = {'original','ANTs img','ANTs enh-np','ANTs enh-p','ANTs den-np','ANTs den-p','ANTs clh-np','ANTs clh-p','Elx img','Elx enh-np','Elx enh-p','Elx den-np','Elx den-p','Elx clh-np','Elx clh-p'};
subdirs = {'original','ants_image_only','ants_enhance_nonpad','ants_enhance_pad','ants_denoise_nonpad','ants_denoise_pad','ants_clahe_nonpad','ants_clahe_pad','elastix_image_only','elastix_enhance_nonpad','elastix_enhance_pad','elastix_denoise_nonpad','elastix_denoise_pad','elastix_clahe_nonpad','elastix_clahe_pad'};
N = 15;

rf   = dir(fullfile(input_dir,'rspace*.mat'));
raw  = load(fullfile(input_dir,rf(1).name),'image');
orig = double(raw.image); orig = orig/max(orig(:));
vm   = sum(orig>=THRESHOLD,4)>=VOTE_THRESH;
pair_idx = nchoosek(1:16,2);

fprintf('\n%-20s  %10s  %10s  %10s  %8s\n','run','PCA_err','temp_ent','NMI_mean','PC1_%');
fprintf('%s\n',repmat('-',1,66));

for r = 1:N
    if r == 1
        img4d = orig;
    else
        s = load(fullfile(comparison_dir,subdirs{r},'gas.mat'),'gas');
        img4d = double(s.gas)/max(double(s.gas(:)));
    end
    [X,Y,Z,T] = size(img4d);

    % PCA error
    e = computePCAError(img4d,N_COMP); e = e.*vm;
    pca_m = mean(e(vm));

    % Temporal entropy
    arr = reshape(img4d,X*Y*Z,T); arr = arr(vm(:),:);
    rmin = min(arr,[],2); rmax = max(arr,[],2);
    arr_n = (arr-rmin)./(rmax-rmin+eps);
    bidx  = min(floor(arr_n*ENT_BINS)+1, ENT_BINS);
    ent   = zeros(size(arr_n,1),1);
    for b = 1:ENT_BINS
        p = sum(bidx==b,2)/T; nz = p>0;
        ent(nz) = ent(nz) - p(nz).*log2(p(nz));
    end
    ent_m = mean(ent);

    % All-pairs NMI
    nmi_v = zeros(120,1);
    for p = 1:120
        A = img4d(:,:,:,pair_idx(p,1));
        B = img4d(:,:,:,pair_idx(p,2));
        a = double(A(vm(:))); b = double(B(vm(:)));
        a = (a-min(a))/(max(a)-min(a)+eps);
        b = (b-min(b))/(max(b)-min(b)+eps);
        edges = linspace(0,1,NMI_BINS+1);
        H2  = histcounts2(a,b,edges,edges);
        H2  = H2/(sum(H2(:))+eps);
        pA  = sum(H2,2); pB = sum(H2,1);
        HA  = -sum(pA(pA>0).*log2(pA(pA>0)));
        HB  = -sum(pB(pB>0).*log2(pB(pB>0)));
        HAB = -sum(H2(H2>0).*log2(H2(H2>0)));
        nmi_v(p) = (HA+HB)/(HAB+eps);
    end
    nmi_m = mean(nmi_v);

    % PC1%
    arr2 = reshape(img4d,X*Y*Z,T); arr2 = arr2(vm(:),:);
    [~,~,~,~,expl] = pca(arr2);
    pc1 = expl(1);

    fprintf('%-20s  %10.5f  %10.5f  %10.5f  %8.2f\n', labels{r}, pca_m, ent_m, nmi_m, pc1);
end
