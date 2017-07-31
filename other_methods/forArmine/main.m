close all
clear all
clc


%% ADDING PATH
addpath(genpath(('./toolboxes/'));
addpath(genpath(('./utils/'));

%% METHOD
METHOD = 'Salzmann09'; %{'Salzmann09','Salzmann11'}

FLATTEMPLATE = true;

%% PARAMS
WITHPLOT = true;

% Armine: I think you will need to play these parameters

% For non-flat template
if ~FLATTEMPLATE
	options.delta.bbs.nC = 20;
	options.delta.bbs.er = 1e-4;
end

switch(METHOD)
	case 'Salzmann09'
		% For Salzmann09
		options.npts_mesh = 20;
		options.phi.nC = 10;
		options.phi.er = 0.55;
	case 'Salzmann11'
		% For Salzmann11
		options.npts_mesh = 20;
		options.phi.nC = 10;
		options.phi.er = 0.55;
		options.NGridx = 20;
		options.NGridy = 20;
end
%% LOADING DATA
temppath = '/home/arvardaz/Dropbox/datasets/templates/paper/texturemap.png'
testimpath = '/home/arvardaz/Dropbox/datasets/paper_100_test/1498837844.6365933_0200.png'




%% RUNNING METHOD
if ~FLATTEMPLATE
	% compute the delta function (flattening function) for non-flat template
	templateData.delta = computeDelta_SfT(templateData, options, WITHPLOT);
end

switch(METHOD)
	case 'Salzmann09'
		output = run_Salzmann09(templateData, inputImageData, options, WITHPLOT)
	case 'Salzmann11'
		output = run_Salzmann11(templateData, inputImageData, options, WITHPLOT)
end