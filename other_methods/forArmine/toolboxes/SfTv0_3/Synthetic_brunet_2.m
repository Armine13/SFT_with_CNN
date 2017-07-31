clear all; close all; clc;
addpath('BBS');
addpath('NLRefinement');
%% load scene
Ncorresp=200;
sigma=0;
scene_ind=2;
exp =1;

load(strcat('savedData/','scene',num2str(scene_ind), ...
        'exp',num2str(exp),'N',num2str(Ncorresp),'sigma_',num2str(sigma),'.mat'));


%% data
K = data.K;
pixels = data.pixels;

p = data.p;   % template (mm)
q = data.q;   % image points (in the camera frame)
GTH = data.P2; % ground truth (mm)

%% some variables
deg2rad = pi/180;
GTHm = 0.001*GTH;
cog = mean(GTHm,2);

RMSE_BRUNET = zeros(1,21);
RMSE_EROL = zeros(1,21);
TICTOC_BRUNET = zeros(1,21);
TICTOC_EROL = zeros(1,21);

counter = 1;

%figure;
%% LOOP
for angle_degree = 0:5:100
    
angle_degree,

%% Perturb ground-truth
non_unit_axis = rand(3,1);
unit_axis = non_unit_axis / norm(non_unit_axis);

theta = angle_degree*deg2rad;
Rot = eye(3)  + sin(theta)*skew(unit_axis) + skew(unit_axis)*skew(unit_axis)*( 1 - cos(theta) );
tr = (angle_degree/100)*norm(cog)*unit_axis;


GTHm0 = GTHm - repmat(cog,[1,size(GTHm,2)]);
GTHm_new = Rot*GTHm0 + repmat( cog + tr, [1,size(GTHm,2)] );

sigma3D = angle_degree/1000;
GTHm_new2=GTHm_new+sigma3D*randn(3,size(GTHm_new,2));


% clf; hold on;
% plot3(GTHm(1,:),GTHm(2,:),GTHm(3,:),'k.');
% plot3(GTHm_new2(1,:),GTHm_new2(2,:),GTHm_new2(3,:),'r.');
% axis equal;
% drawnow;
% % 
% % 
% pause;

GTH_perturbed = 1000*GTHm_new2;  % mm    



%% Brunet's method
options.phi.er = 1e1;
options.maxiter = 10;
options.phi.nC = 24;
options.phi.er = 1e0;
options.lbdiso = 1e0;
options.verbose = 1;
options.lbd_inext= 50;
options.lbd_bend=50;

% Run Brunet's method on P2:
tic,
out = pts_to_brunet( GTH_perturbed, p, q, options );
shape_brunet = bbs_eval(out.phi.bbs,out.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
tictoc_brunet = toc,

rmse_brunet = sqrt(mean(sum( shape_brunet - GTH ).^2)),


RMSE_BRUNET(counter) = rmse_brunet;
TICTOC_BRUNET(counter) = tictoc_brunet;

%% Erol's part
options.eta.er = 1e2;
options.phi.er = 1e1;
options.eta.nC = 24;
options.phi.nC = 24;
options.K = K;

tic,
griddata = pts_to_erol(GTH_perturbed, p, q, options);  % generate regular grid data

template_grid = 0.001*griddata.template; % meters
pixels_grid = griddata.pixels;
GTH_perturbed_grid = 0.001*griddata.gth;
template_grid(3,:) = 0.25;

% SfT
moptions.K = K;               % intrinsic camera matrix
moptions.mass = 0.01;         % template weight (kg)
moptions.k_damping = 0.008;   % template velocity damping coefficient between [0...1]  
moptions.k_stretch = 1;       % template stretching stiffness coefficient between [0...1] (isometric case = 1)
moptions.k_bend = 1;          % template bending stiffness coefficient between [0...1] 
moptions.maxIter = 1000;      % maximum number of loop iterations (default 2500)
moptions.solverIter = 1;      % number of solver iterations 
moptions.threshold = 1e-6;   % minimum displacement threshold to stop the loop  (default rmse 10e-6) 


shape_grid = SfTerol( GTH_perturbed_grid, template_grid, pixels_grid, moptions );
%shape_grid = SfTerol( template_grid, template_grid, pixels_grid, moptions );

% go back to original pts:
template_grid = griddata.template; % meters
shape_grid = 1000*shape_grid;

[bbs_phi, ctrlpts_phi] = make_bbs_warp(template_grid, shape_grid, options.phi.nC, options.phi.er);
shape_erol = bbs_eval(bbs_phi,ctrlpts_phi,p(1,:)',p(2,:)',0,0);
tictoc_erol = toc,


rmse_erol = sqrt(mean(sum( shape_erol - GTH ).^2)),

RMSE_EROL(counter) = rmse_erol;
TICTOC_EROL(counter) = tictoc_erol;


counter = counter + 1;

end

save('basin','RMSE_BRUNET','RMSE_EROL','TICTOC_BRUNET','TICTOC_EROL');

x = 0:5:100;
figure; hold on;
plot(x,RMSE_BRUNET,'r');
plot(x,RMSE_EROL,'k');
title('3D error');

figure; hold on;
plot(x,TICTOC_BRUNET,'r');
plot(x,TICTOC_EROL,'k');
title('Time');
