% IPPE method for CVLAB dataset
clear all; close all; clc;
addpath(genpath('../ExtLibs/utilsmatlab'));
addpath('NLRefinement/');
addpath('BBS');
addpath('../ExtLibs/ris/');
addpath('../ExtLibs/cvx/sedumi/');
addpath('../ExtLibs/cvx/sdpt3/');
addpath('../ExtLibs/SALZ/');
%%
load('ext_varying_corresp100_sigma1_bound20.mat');
RMSE = [];

tic,
for i = 1:50
i,
ground_truth = 1000*gth{i}; % metric units in the camera frame (mm)
pixels = imgpts{i};  % pixels
template = 1000*tpl{i};
bii = bi{i};


p = template(1:2,:);  
n_particles = size(p,2);    % number of particles
fx = K(1,1);  fy = K(2,2);
uo = K(1,3);  vo = K(2,3);
m = pixels - repmat( [uo;vo], [1, n_particles] );
q = m ./ repmat( [fx;fy], [1, n_particles] ); 


    
    %% BGCC12C method
    options.eta.er = 5e1;
    options.eta.nC = 12;
    options.phi.er = 5e2;
    options.phi.nC = 12;
    options.maxiter = 100;
    options.verbose = 0;
    options.method = 'BGCC12C'; 
    


    outd = SfT(p,q,options);    
     
    rmsev = [];
    for s = 1:length(outd.phi)
        Qd = bbs_eval(outd.phi{s}.bbs,outd.phi{s}.ctrlpts,p(1,:)',p(2,:)',0,0);
        [Qd,alpha,signo]=RegisterToGTH(Qd,ground_truth);
        rmsev(s) = sqrt(mean(sum(Qd-ground_truth).^2));
    end
 
    
   [rmse, rmsei] = min(rmsev),
    

    shape = bbs_eval(outd.phi{rmsei}.bbs,outd.phi{rmsei}.ctrlpts,p(1,:)',p(2,:)',0,0);
    [shape,alpha,signo]=RegisterToGTH(shape,ground_truth);
    rmse = sqrt(mean(sum(shape-ground_truth).^2)),

    RMSE = [RMSE,rmse];
end
toc,

save('./rmse_BGCC12C_ext_varying_with20bp','RMSE');


figure;
plot(RMSE);
axis([1 50 0 50]);

figure,
plot3(shape(1,:),shape(2,:),shape(3,:),'bo'); hold on;
plot3(ground_truth(1,:),ground_truth(2,:),ground_truth(3,:),'go');
axis equal; hold off;
