function [ out ] = pts_to_brunet( P3, p, q, options, tmplroi )
% Brunet's method with 3D points

% generate grid proi
umin = min(p(1,:)); umax = max(p(1,:)) ;
vmin = min(p(2,:)); vmax = max(p(2,:)) ;
options.KLims = [umin umax vmin vmax];
options.NGridx = 20; options.NGridy = 20;

NRoi=options.NGridy*options.NGridx;
[xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
proi=[xroi(:)';yroi(:)'];

if nargin == 5   
    bd = bwboundaries(tmplroi,'noholes');
    % x and y points of the boundary
    boundary = bd{1};
    xi = boundary(:,2);
    yi = boundary(:,1);
    p1 = options.K*[proi;ones(1,length(proi))];
    onmat = inpolygon(p1(1,:),p1(2,:),xi,yi);
    proi = [proi(1,onmat~=0);proi(2,onmat~=0)];    
end


[bbs, ctrlpts] = make_bbs_warp(p, P3, options.phi.nC, options.phi.er);

% out = NLRefinebbs(bbs,ctrlpts,options,p,q,proi,[],[]);
lbd_inext = options.lbd_inext;
lbd_bend = options.lbd_bend;
out = NLrefine_ris( bbs, ctrlpts, p, q, lbd_inext, lbd_bend);

end
