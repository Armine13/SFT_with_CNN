clear all; close all;
addpath('BBS');
addpath('NLRefinement');
% addpath('utils');

% Number of points detected
N=200;
sigma=1;

% Load scene

eval(sprintf('load Escena5'));

% scene 3D points
P=scene.Msh.vertexPos';
% camera calibration  
K=scene.camera.params.K;
T=scene.camera.params.T;
R=scene.camera.params.R;

% Project points
P2=R*P+T*ones(1,size(P,2));
m2=K*(P2);
m2=m2./([1;1;1]*m2(3,:));
% Add noise
m2(1:2,:)=m2(1:2,:)+sigma.*randn(2,size(P,2));
m3=inv(K)*m2;
q=m3(1:2,:);
% Points in the template
I=scene.Msh.texMap.img;
xgth=1+scene.Msh.texMap.vertexUVW(:,1)*size(I,2);
ygth=size(I,1)-scene.Msh.texMap.vertexUVW(:,2)*size(I,1)+1;
p=[xgth';ygth'];
% Select N random points
indices=randi(size(P,2),1,N);
% Show features in the template and input
I2=scene.render.imOut;


options.eta.er = 1e3;
options.eta.nC = 28;
options.phi.er = 1e3;

options.maxiter = 10;

% brunet
options.phi.nC = 24;
options.phi.er = 1e0;
options.lbdiso = 1e0;
options.verbose = 1;

options.lbd_inext= 50;
options.lbd_bend=200;

options.K = K;

p = p(:,indices);
q = q(:,indices);
P2 = P2(:,indices);

% Run Brunet's method on P2:
% P2 is the 3D points GTH 
out = pts_to_brunet(P2,p,q,options);
Qdr = bbs_eval(out.phi.bbs,out.phi.ctrlpts,p(1,:)',p(2,:)',0,0);

disp(sprintf('Reconstruction error = %f',sqrt(mean(sum(Qdr-P2).^2))));

