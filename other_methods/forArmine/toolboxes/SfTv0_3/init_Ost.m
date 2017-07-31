%% Initialize

rootpath = '../laplacianmeshes';
% assert(strcmp(rootpath((end-14):end), 'laplacianmeshes'));

addpath(genpath(rootpath));
% mex(fullfile(rootpath, 'LengthConstraints.c'));

