% version

% Shape from Template example with synthetic data
%   This code is partially based on the work of  
%   [Bartoli et. al 2012] On Template-Based Reconstruction from a Single View:
%   Analytical Solutions and Proofs of Well-Posedness for  Developable,
%   Isometric and Conformal Surfaces
%   (c) 2013, Adrien Bartoli and Daniel Pizarro. dani.pizarro@gmail.com 
%
% SfT is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% SfT is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
% for more details.
% 
% You should have received a copy of the GNU General Public License along
% with this program; if not, write to the Free Software Foundation, Inc.,
% 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
clear all;
close all;

addpath('BBS');
addpath('NLRefinement/');
addpath('../ExtLibs/ris/');
addpath('../ExtLibs/cvx/sedumi/');
addpath('../ExtLibs/cvx/sdpt3/');
addpath('../ExtLibs/SALZ/');
addpath('../ExtLibs/utilsmatlab/');

% scenes validated: 1,2,3,4,5,6,9,11,12,13
scenes = [1 2 3 4 5 6 9 11 12 13];

% Load scene
focal_ind = 500;
s = 1;
Ncorresp = 100;    
sigma = 1;
scene_ind = 1;
load scene1s-factor1exp1N100sigma1;    
p = data.p;
q = data.q;
P2 = data.P2;
indices = data.indices;

% Surface reconstruction parameters for BBS:
options.eta.er = 5e2;
options.eta.nC = 24;
options.phi.er = 8e1;
options.phi.nC = 24;
options.maxiter = 10;

p = p(:,indices);
q = q(:,indices);
P2 = P2(:,indices);

% Create Warp and the derivative:
er = options.eta.er;
nC = options.eta.nC;

% Make BBS warp and differentiate
umin = min(p(1,:)) -0.05; umax = max(p(1,:)) +0.05;
vmin = min(p(2,:)) -0.05; vmax = max(p(2,:)) +0.05;
options.KLims = [umin umax vmin vmax];
options.NGridx = 50; options.NGridy = 50;
[bbs, ctrlpts] = make_bbs_warp(p,q,nC,er,options.KLims);

NRoi=options.NGridy*options.NGridx;
[xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
proi=[xroi(:)';yroi(:)'];

Jwu = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',1,0);
Jwv = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,1);
Jws = [Jwu;Jwv];
qw = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,0);

% Apply IPPE and get the resulting embedding on a dense grid
Jw = [Jws(1,:);Jws(3,:);Jws(2,:);Jws(4,:)];
[Q, N1, N2] = IPPEO(qw, Jw);

% Build a BBS warp and obtain the embedding on the original feature points
[bbs3,ctrlpts3] = make_bbs_warp(proi,Q,nC,options.phi.er,options.KLims);
Qw = bbs_eval(bbs3,ctrlpts3,p(1,:)',p(2,:)',0,0);

% Find embedding normal field
nC = 28;
er = 2e-3;
umin = min(qw(1,:)) -0.05; umax = max(qw(1,:)) +0.05;
vmin = min(qw(2,:)) -0.05; vmax = max(qw(2,:)) +0.05;
optionsq.KLims = [umin umax vmin vmax];
[bbsn, ctrlptsn] = make_bbs_warp(qw,Q,nC,er,optionsq.KLims);
Nd = construct_normals(bbsn,ctrlptsn,qw);

% Find the ground truth embedding normal field
nC = 28;
umin = min(qw(1,:)) -0.05; umax = max(qw(1,:)) +0.05;
vmin = min(qw(2,:)) -0.05; vmax = max(qw(2,:)) +0.05;
optionsq.KLims = [umin umax vmin vmax];
[bbsg, ctrlptsg] = make_bbs_warp(q,P2,nC,er,optionsq.KLims);
Ng = construct_normals(bbsg,ctrlptsg,qw);

% Find a single normal field
N = disambiguate_normals(N1,N2,Nd);

% Find score for the analytical and constructed normals
Cscore = zeros(length(N),1);
Ascore = zeros(length(N),1);
for i = 1: length(N)
    Cscore(i) = Ng(:,i)'*Nd(:,i);
    Ascore(i) = Ng(:,i)'*N(:,i);
end
Cscore = sum(Cscore)/length(N)
Ascore = sum(Ascore)/length(N)

% Do shape from normals to get the embedding
pt = [qw; ones(1,size(qw,2))];
nC=28;
lambdas = 10e-3*ones(nC-3, nC-3);
bbsd = bbs_create(umin, umax, nC, vmin, vmax, nC, 1);
colocd = bbs_coloc(bbsd, pt(1,:), pt(2,:));
bendingd = bbs_bending(bbsd, lambdas);
[ctrlpts3Dn]=ShapeFromNormals(bbsd,colocd,bendingd,pt,N);
ctrlpts3Dn = real(ctrlpts3Dn);
mu=bbs_eval(bbsd, ctrlpts3Dn, pt(1,:)', pt(2,:)',0,0);
% Correct integration constant:
% mu = mu *(median(Qw(3,:))/median(mu));
Qd = [qw(1,:).*mu; qw(2,:).*mu; mu];

[bbs3,ctrlpts3] = make_bbs_warp(proi,Qd,options.phi.nC,options.phi.er,options.KLims);
Qn = bbs_eval(bbs3,ctrlpts3,p(1,:)',p(2,:)',0,0);
Qn = RegisterToGTH(Qn,Qw);
[bbsn, ctrlptsn] = make_bbs_warp(p,Qn,options.phi.nC,options.phi.er,options.KLims);
%% SfT's for the same data:
% Surface reconstruction parameters for BBS:
options.eta.er = 5e2;
options.eta.nC = 24;
options.phi.er = 8e1;
options.phi.nC = 24;
options.maxiter = 10;

options.method = 'BGCC12I';
outd = SfTJbbs(p,q,options);

options.phi.er = 8e1;
options.eta.er = 5e2;

options.method = 'CPB14I';
outj = SfTJbbs(p,q,options);
options.eta.er = 5e2;

% outjr = NLRefinebbs(outj.phi.bbs,outj.phi.ctrlpts,options,p,q,proi,[],[]);
Qd = bbs_eval(outd.phi.bbs,outd.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
Qj = bbs_eval(outj.phi.bbs,outj.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
% Qjr = bbs_eval(outjr.phi.bbs,outjr.phi.ctrlpts,p(1,:)',p(2,:)',0,0);

disp(sprintf('Reconstruction error AnD = %f',sqrt(mean(sum(Qd-P2).^2))));
% disp(sprintf('Reconstruction error direct after refinement = %f',sqrt(mean(sum(Qdr-P2).^2))));
disp(sprintf('Reconstruction error AnJ = %f',sqrt(mean(sum(Qj-P2).^2))));
% disp(sprintf('Reconstruction error AnJ after refinement = %f',sqrt(mean(sum(Qjr-P2).^2))));
% disp(sprintf('Reconstruction error AnJ refined = %f',sqrt(mean(sum(Qjr-P2).^2))));
disp(sprintf('pause'));
%%
options.phi.er = 8e1;
options.eta.er = 5e2;
options.outlierreject = 'none';
options.planar = true;
options.K = [500*(s+1) 0 320; 0 500*(s+1) 240; 0 0 1];
options.method = 'Ostlund';
outo = SfTJbbs(p,q,options);

options.method = 'Salz2';
outs = SfTJbbs(p,q,options);

Qs = bbs_eval(outs.phi.bbs,outs.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
Qo = bbs_eval(outo.phi.bbs,outo.phi.ctrlpts,p(1,:)',p(2,:)',0,0);

disp(sprintf('Reconstruction error Salzmann = %f',sqrt(mean(sum(Qs-P2).^2))));
disp(sprintf('Reconstruction error Ostlund = %f',sqrt(mean(sum(Qo-P2).^2))));

disp(sprintf('Reconstruction from normals error = %f',sqrt(mean(sum(Qn-P2).^2))));