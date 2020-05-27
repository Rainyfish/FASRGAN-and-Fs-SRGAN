%% Parameters
% Directory with your results
%%% Make sure the file names are as exactly %%%
%%% as the original ground truth images %%%
% GT and SR folder
folder_GT = '../Results/HR/Set5';
folder_SR = '../Results/SR/Set5';

method = {'FASRGAN', 'Fs-SRGAN', 'FA+Fs-SRGAN'};

num_method = length(method);

% Number of pixels to shave off image borders when calcualting scores
shave_width = 0;    

% Set verbose option
verbose = true;

%% Calculate scores and save

for idx_method = 1:num_method
    fprintf([method{idx_method}, ' is processing']);
    fprintf('\n');
    GT_dir = folder_GT ;
    input_dir  = fullfile(folder_SR, [method{idx_method}]);
    % calculate PI
    addpath utils
    scores = calc_scores(input_dir,GT_dir,shave_width,verbose);
    % Printing results
    perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;
    fprintf(['\n\nYour perceptual score is: ',num2str(perceptual_score)]);
    fprintf(['\nYour RMSE is: ',num2str(sqrt(mean([scores.MSE]))),'\n']);
    end

%
