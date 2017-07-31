function [ griddata ] = pts_to_erol( P3, p, q, options )
% Brunet's method with 3D points

% generate template grid
umin = min(p(1,:)); umax = max(p(1,:)) ;
vmin = min(p(2,:)); vmax = max(p(2,:)) ;
options.KLims = [umin umax vmin vmax];
options.NGridx = 20; options.NGridy = 20;

NRoi=options.NGridy*options.NGridx;
[xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
proi=[xroi(:)';yroi(:)']; % template grid

% template-image warp: pixels grid
[bbs_eta, ctrlpts_eta] = make_bbs_warp(p, q, options.eta.nC, options.eta.er);
qroi = bbs_eval(bbs_eta,ctrlpts_eta,proi(1,:)',proi(2,:)',0,0);
qroi_pix = options.K*[qroi; ones(1,length(qroi))];
pixels_grid = qroi_pix(1:2,:) ./ ([1;1]*qroi_pix(3,:)); % image grid

% template - groundtruth warp:  groundtruth grid 
[bbs_phi, ctrlpts_phi] = make_bbs_warp(p, P3, options.phi.nC, options.phi.er);
P3_grid = bbs_eval(bbs_phi,ctrlpts_phi,proi(1,:)',proi(2,:)',0,0);


% data to return:
griddata.template = proi;
griddata.pixels = pixels_grid;
griddata.gth = P3_grid;


end
