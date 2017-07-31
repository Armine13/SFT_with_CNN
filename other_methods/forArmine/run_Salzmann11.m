function output = run_Salzmann11(templateData, inputImageData, options, WITHPLOT)
% This function runs the method of Salzmann11 with a flat or non-flat template
% M. Salzmann and P. Fua. Linear local models for monocular reconstruction of deformable surfaces.
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(5), May 2011. 3, 7, 11, 13, 16, 19, 21

% INPUTS:
% templateData.crsp.imgPts_mm: correspondences in the texture-map, expressed in mm
% templateData.crsp.imgPts_px: correspondences in the texture-map, expressed in px
% inputImageData.crsp.imgPts_px: correspondences in the input image, expressed in px
% inputImageData.crsp.camPtsGT_mm: 3D correspondences visible in the input image, expressed in mm
% inputImageData.KK: intrinsic parameters
% options:
%			options.npts_mesh: number of mesh points to initialize the method (default 20)
% 			options.phi.nC: number of control points (default 10)
% 			options.phi.er: smoothing parameter (default 0.55)
% 			options.NGridx: number of grid points for x-axis (default 15)
% 			options.NGridy: number of grid points for y-axis (default 15)

% OUTPUTS:
% output.phi.Q: 3D correspondences estimated by the method of Salzmann, expressed in mm
% output.inputImageData.crsp.camPts_mm: 3D estimated correspondences fitted with a B-spline, expressed in mm (the best to choose)
% output.phi.bbs: B-sline data
% output.phi.ctrlpts: B-sline control points


p = templateData.crsp.imgPts_mm;
q = inputImageData.crsp.imgPts_px;


umin=min(p(1,:));
umax=max(p(1,:));
vmin=min(p(2,:));
vmax=max(p(2,:));

margin_u = umax - umin;
margin_v = vmax - vmin;

umin = umin - 0.05*margin_u;
umax = umax + 0.05*margin_u;
vmin = vmin - 0.05*margin_v;
vmax = vmax + 0.05*margin_v;


KLims=[umin,umax,vmin,vmax];

%%
if isfield(templateData,'delta') % the template is not flat
	[px, py] = meshgrid(linspace(KLims_n(1), KLims_n(2), options.npts_mesh), linspace(KLims_n(3), KLims_n(4), options.npts_mesh));
	P3d = bbs_eval(templateData.delta.bbs,templateData.delta.ctrlpts,px,py,0,0);


	% Convert the template points into barycentric coordinates
	[tri, ~, ~, ~] = ris_create_tri_mesh(KLims_n(1), KLims_n(2), options.npts_mesh, KLims_n(3), KLims_n(4), options.npts_mesh);
	tri_x = P3d(1,:)'; tri_y = P3d(2,:)'; tri_z = P3d(3,:)';
	P3 = bbs_eval(templateData.delta.bbs,templateData.delta.ctrlpts,p(1,:),p(2,:),0,0);            

	[b1, b2, b3, ind_tri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z, P3(1,:), P3(2,:), P3(3,:)); 
else % the template is flat
	[tri tri_x tri_y tri_z] = ris_create_tri_mesh(KLims(1), KLims(2), options.npts_mesh, KLims(3), KLims(4), options.npts_mesh);

	% Convert the template points into barycentric coordinates
	[b1 b2 b3 ind_tri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z,p(1,:),p(2,:), ones(size(p(2,:))));
end 

meshIn.vertexPos=[tri_x,tri_y,tri_z];
meshIn.faces=tri;


[meshOut,Q] = SalzReconstructionL(q(1:2,:), ind_tri, [b1;b2;b3]', meshIn, inputImageData.KK, options.NGridx, options.NGridy); % here the only difference with 'run_Salzmann11.m'


bbs = bbs_create(umin, umax, options.phi.nC, vmin, vmax, options.phi.nC, 3);

coloc = bbs_coloc(bbs, p(1,:), p(2,:));
lambdas = options.phi.er*ones(options.phi.nC-3, options.phi.nC-3);
bending = bbs_bending(bbs, lambdas);

% get control points for i to j warp
cpts = (coloc'*coloc + bending) \ (coloc'*Q(1:3,:)');
ctrlpts = cpts';

Qw = bbs_eval(bbs,ctrlpts,p(1,:)',p(2,:)',0,0);

%Visualize Point Registration Error
error = sqrt(mean((Qw(1,:)-Q(1,:)).^2+(Qw(2,:)-Q(2,:)).^2+(Qw(3,:)-Q(3,:)).^2));
disp([sprintf('[PHI] Internal Rep error = %f',error)]);

output.phi.Q = Q;
output.inputImageData.crsp.camPts_mm = Qw;
output.phi.bbs = bbs;
output.phi.ctrlpts = ctrlpts;


%% %%%%%%%%%%%%%%%%%%%%%%%%%% DISPLAY
if WITHPLOT
    figure(20);
    clf;
	subplot(221);    
    plot3(P3d(1,:),P3d(2,:),P3d(3,:),'r+');
    axis equal;
    title('Delta embedding function applied to a grid');
    subplot(222);    
    plot3(P3(1,:),P3(2,:),P3(3,:),'r+');
    axis equal;
    title('Delta embedding function applied to barycentric coordinates');
    subplot(223);
    imshow(templateData.texturemap);
    hold on;
    plot(templateData.crsp.imgPts_px(1,:),templateData.crsp.imgPts_px(2,:),'c+');
    hold off;
    title('Correspondences on the texture-map');
    subplot(224);
    plot3(inputImageData.crsp.camPtsGT_mm(1,:),inputImageData.crsp.camPtsGT_mm(2,:),inputImageData.crsp.camPtsGT_mm(3,:),'r.');
    hold on;
    plot3(output.inputImageData.crsp.camPts_mm(1,:),output.inputImageData.crsp.camPts_mm(2,:),output.inputImageData.crsp.camPts_mm(3,:),'go');
    hold off;
    legend('GT crsp',['Estimated crsp using Salzmann11']);
    axis equal;
    title('Reconstruction of the correspondences')
end