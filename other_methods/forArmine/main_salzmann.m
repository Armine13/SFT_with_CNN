close all
clear all
clc


%% ADDING PATH
addpath(genpath('./toolboxes/'))
addpath(genpath('./utils/'));

%% METHOD
METHOD = 'Salzmann09';%'Chhatkuli17'; %{'Salzmann09','Salzmann11','Chhatkuli17'}

FLATTEMPLATE = true;

%% PARAMS
WITHPLOT = false;

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
temppath = '/home/arvardaz/Dropbox/datasets/templates/paper/texturemap.png';
testimpath = '/home/arvardaz/Dropbox/datasets/paper_100_test/';
imname = '1498837844.6365933_0200.png';
[~,fname,~] = fileparts(imname);


gaimDir = fileparts(which('GAIM_example'));
workDir = [gaimDir '/workDir/']; %current directory is used as the working directory (for saving intermediate results).
if ~exist(workDir,'dir')
   mkdir(workDir); 
end

setGAIMDependencies();
% 
detectorOpts = parseDetectOpts(struct);
% 
matchOpts = parseMatchOpts(struct,detectorOpts);
deleteTemFiles = 0;
verb = 1;
[crspTemplate_px,crspImg_px] = GAIM_matcher(temppath,strcat(testimpath,imname),workDir,detectorOpts,matchOpts,deleteTemFiles,verb,WITHPLOT);


%[crspTemplate_px,crspImg_px] = cpselect(imread(temppath), imread(strcat(testimpath,imname)),'Wait',true);
% load('crspTemplate_px.mat');
% load('crspImg_px.mat');
% crspTemplate_px = crspTemplate_px';
% crspImg_px = crspImg_px';

% templateData.crsp.imgPts_mm: correspondences in the texture-map, expressed in mm
% templateData.crsp.imgPts_px: correspondences in the texture-map, expressed in px
% inputImageData.crsp.imgPts_px: correspondences in the input image, expressed in px
% inputImageData.crsp.camPtsGT_mm: 3D correspondences visible in the input image, expressed in mm
% inputImageData.KK: intrinsic parameters

meshInputImage = read3dMesh(strcat(testimpath, fname, '.obj'));
meshInputImage.vertexPos = meshInputImage.vertexPos;
d = load(strcat(testimpath, fname, '.csv'));
f = d(2);
gt_coords = d(end-3005:end);
gt_coords = reshape(gt_coords, 3, 1002);

K = [f, 0, 112; 0, f, 112; 0, 0, 1];

templateData.texturemap = imread(temppath);
templateData.scale_mm2px =  size(templateData.texturemap,1)/1.6637553648373311;
% templateData.scale_mm2px = size(templateData.texturemap,1)/3.328;

templateData.crsp.imgPts_mm = crspTemplate_px/templateData.scale_mm2px;
templateData.crsp.imgPts_px = crspTemplate_px;
templateData.crsp.camPts_mm = templateData.crsp.imgPts_mm;
templateData.crsp.camPts_mm(3,:) = 0;
% inputImageData.texturemap = imread(strcat(testimpath,imname));
inputImageData = struct('crsp',struct('imgPts_px',crspImg_px,'imgPtsGT_mm',crspImg_px), 'KK', K);
imgPts_px_temp = inputImageData.crsp.imgPts_px;
imgPts_px_temp(3,:) = 1;
inputImageData.crsp.imgPts_n = inv(K)*imgPts_px_temp;


inputImageData.crsp.camPtsGT_mm = extract3DCrsp_from2DCrsp_forMesh(imread(temppath), imread(strcat(testimpath,imname)), K, meshInputImage, templateData.scale_mm2px, crspTemplate_px);%4.8363, crspTemplate_px);

% inputImageData.crsp.camPtsGT_mm = inputImageData.crsp.camPtsGT_mm(:,1:5:end);
% inputImageData.crsp.imgPtsGT_mm = inputImageData.crsp.imgPtsGT_mm(:,1:5:end);
% inputImageData.crsp.imgPts_px = inputImageData.crsp.imgPts_px(:,1:5:end);
% templateData.crsp.imgPts_mm = templateData.crsp.imgPts_mm(:,1:5:end);
% templateData.crsp.imgPts_px = templateData.crsp.imgPts_px(:,1:5:end);

% templateData.crsp.imgPts_mm = templateData.crsp.imgPts_mm/20;
% templateData.crsp.imgPts_px = templateData.crsp.imgPts_px/20;
% templateData.texturemap = imresize(templateData.texturemap,1/20);


options.npts_mesh = 20;
options.phi.nC = 16;
options.phi.er = 1e-1;

%% RUNNING METHOD
if ~FLATTEMPLATE
	% compute the delta function (flattening function) for non-flat template
	templateData.delta = computeDelta_SfT(templateData, options, WITHPLOT);
end

switch(METHOD)
	case 'Salzmann09'
		output = run_Salzmann09(templateData, inputImageData, options, WITHPLOT);
	case 'Salzmann11'
		output = run_Salzmann11(templateData, inputImageData, options, WITHPLOT);
    case 'Chhatkuli17'    
        output = run_Chhatkuli17(templateData, inputImageData, options, WITHPLOT);
end
%%
% Define p (=mesh vertices?)
% p2 = reshape(gt_coords, 3, 1002);
% p = p2./repmat(p2(3,:),3,1) ;
p = uv2PixCoords(meshInputImage.texMap.vertexUVW(:,1:2),size(templateData.texturemap,2),size(templateData.texturemap,1));
p = p';
p = p./templateData.scale_mm2px;
meshVertices_reconstructed = bbs_eval(output.phi.bbs,output.phi.ctrlpts,p(1,:)',p(2,:)',0,0);%*templateData.scale_mm2px;
meshVertices_reconstructed = 2*meshVertices_reconstructed;

rmse = sqrt(mean(mean((gt_coords - meshVertices_reconstructed).^2)))

%%
figure(53)
clf;
% 
plot3(meshVertices_reconstructed(1,:), meshVertices_reconstructed(2,:),meshVertices_reconstructed(3,:),'b*');
hold on;

plot3(gt_coords(1,:), gt_coords(2,:),gt_coords(3,:),'r*');
% plot3(gt_coords(1,:), gt_coords(2,:),gt_coords(3,:),'r*');
 axis equal;
