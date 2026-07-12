% run_CompReg.m
% Compare any two gas image series using CompReg.
% Set seriesA and seriesB below — each is a struct with 'dir', 'stage', 'label'.
%
% Available dirs:   'test_543200iter'  (iterations [500,400,300,200])
%                   'test_1000iter'    (iterations [1000])
%                   'test_5432000iter' (iterations [5000,4000,3000,2000])
%                   'test_5000iter'    (iterations [5000], no ANTs)
% Available stages: 'final_groupwise' | 'final_ants'
% label: any valid MATLAB identifier (no spaces/brackets)

cd('/Users/hoomham/Hooman/Work/Codes/2026_PCA_registration/');

base = '/Volumes/HoomHamExt/Work/Analysis/2024-11-13_025JC/reg/';

% --- Edit these two lines to pick what you compare ---
seriesA = struct('dir', 'test_543200iter',    'stage', 'final_groupwise', 'label', '543200');
seriesB = struct('dir', 'test_1000iter',    'stage', 'final_groupwise', 'label', '1000');
  
% groupwise vs ANTs, same run
% seriesA = struct('dir','test_1000iter', 'stage','final_groupwise', 'label','gw_1000');
% seriesB = struct('dir','test_1000iter', 'stage','final_ants',      'label','ants_1000');
% 
% % two groupwise iterations
% seriesA = struct('dir','test_1000iter',    'stage','final_groupwise', 'label','gw_1000');
% seriesB = struct('dir','test_5000iter',    'stage','final_groupwise', 'label','gw_5000');
% 
% % two ANTs runs
% seriesA = struct('dir','test_543200iter',  'stage','final_ants', 'label','ants_500_400');
% seriesB = struct('dir','test_5432000iter', 'stage','final_ants', 'label','ants_5000_4000');

% ------------------------------------------------------
% Other examples:
%   groupwise vs ANTs, same run:
%     seriesA = struct('dir','test_1000iter', 'stage','final_groupwise', 'label','gw_1000');
%     seriesB = struct('dir','test_1000iter', 'stage','final_ants',      'label','ants_1000');
%   two ANTs runs:
%     seriesA = struct('dir','test_543200iter',  'stage','final_ants', 'label','ants_500_400');
%     seriesB = struct('dir','test_5432000iter', 'stage','final_ants', 'label','ants_5000_4000');

volA = double(abs(niftiread(fullfile(base, seriesA.dir, seriesA.stage, 'gas.nii'))));
volB = double(abs(niftiread(fullfile(base, seriesB.dir, seriesB.stage, 'gas.nii'))));

img        = cat(5, volA, volB);
imlabels   = {seriesA.label, seriesB.label};
thresholds = [0.04, 0.04];
visualizeFlag = false;

CompReg(img, imlabels, thresholds, visualizeFlag);
