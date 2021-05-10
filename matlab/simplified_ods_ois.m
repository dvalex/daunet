function [sODS, sOIS] = simplified_ods_ois(ResultsDir, Dataset, Method)
%% [sODS, sOIS] = SIMPLIFIED_ODS_OIS(RESULTSDIR, DATASET, METHOD)
% Computes simplified ODS/OIS measures proposed in paper 
% "DAUNet: Deep Augmented Neural Network for Pavement Crack Segmentation"
% Parmaeters:
% ResultsDir - segmentation relults are located
% Dataset - dataset name 
% Method - method name
% Supposed directory strucure:
% ResultsDir
%   |----Dataset
%           |----Method (directory with segmentation results)
%           |----gt     (directory with ground truth)
if nargin == 0
    ResultsDir = '..\..\RESULTS';
    Dataset ='CRACK500';
    Method = 'FPHBN';%'DAUNet';
end

pred_dir = fullfile(ResultsDir, Dataset, Method); %directory with predictions
gt_dir = fullfile(ResultsDir, Dataset, 'gt'); % ground truth


eval_dir = [pred_dir '_eval1'];
mkdir(eval_dir);
gt_files = dir(fullfile(gt_dir, '*.png'));
N = length(gt_files);
thresholds = 0.01:0.01:0.99;
n_th = length(thresholds);
run_stage1 = true;
if run_stage1 % calculate P-R for all images in dataset
    tic;
    parfor i = 1:N
        gtname = gt_files(i).name;

        gtmask = get_mask(fullfile(gt_dir, gtname));
        allGT = sum(gtmask(:));

        predname = strrep(gtname, 'gt', 'pr');
        C= zeros(n_th,1);
        PRED = zeros(n_th, 1);
        TP  = zeros(n_th, 1);
        for t = 1:n_th
            predmask = get_mask(fullfile(pred_dir, predname), thresholds(t));
            PRED(t) = sum(predmask(:));
            matchmask = gtmask & predmask;
            TP(t) = sum(matchmask(:));
        end
        [~, basegtname,~]= fileparts(gtname);
        fnameout = fullfile(eval_dir,  [ basegtname, '_ev1.txt']);
        fID = fopen(fnameout, 'wt'); 
        for t = 1:n_th
            fprintf(fID, '%g\t%d\t%d\t%d\t%d\n', thresholds(t), TP(t), allGT, TP(t), PRED(t));
        end
        fclose(fID);
    end
    toc;
end

%% stage 2 - collect from files
PR_ACCUM = zeros(n_th, 4);
feval = dir (fullfile(eval_dir, '*_ev1.txt'));
assert(length(feval)==N);
Farr = zeros(N,1);
for i = 1:N
    evname = feval(i).name;
    evdata = load(fullfile(eval_dir, evname));
    assert(size(evdata,1) == n_th && size(evdata,2)==5);
    PR_ACCUM  = PR_ACCUM  + evdata(:, 2:end);
    
    R_i = evdata(:,2) ./ (evdata(:,3)+eps);
    P_i = evdata(:,4) ./ (evdata(:,5)+eps);
    F_i = 2*P_i .* R_i ./ (P_i+R_i+eps);
    Fmax = max(F_i);
    Farr(i)=Fmax;
end

sOIS = mean(Farr);
R_ds = PR_ACCUM(:,1) ./ PR_ACCUM(:,2);
P_ds = PR_ACCUM(:,3) ./ PR_ACCUM(:,4);
F_ds = 2*P_ds .* R_ds ./ (P_ds+R_ds);
sODS = max(F_ds);
fprintf('%s %s:\tsODS = %g\tsOIS=%g\n',Dataset, Method, sODS, sOIS);




        
