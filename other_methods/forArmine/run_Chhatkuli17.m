function output = run_Chhatkuli17(templateData, inputImageData, options, WITHPLOT)
% This function runs the method ofchhatkuli17

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

% OUTPUTS:
% output.phi.Q: 3D correspondences estimated by the method of Salzmann, expressed in mm
% output.inputImageData.crsp.camPts_mm: 3D estimated correspondences fitted with a B-spline, expressed in mm (the best to choose)
% output.phi.bbs: B-sline data
% output.phi.ctrlpts: B-sline control points
%

%% %%%%%%%%%%%%%%%%%%%%%%%%%% SfT PARAMS
% options.method='CPB14IR';
options.method='CPBC15I';

umin = 0;
umax = size(templateData.texturemap,2)./templateData.scale_mm2px;
vmin = 0;
vmax = size(templateData.texturemap,1)./templateData.scale_mm2px;
options.KLims = [umin,umax,vmin,vmax];

if mean(templateData.crsp.camPts_mm(3,:)) == 0
    %% %%%%%%%%%%%%%%%%%%%%%%%%%% RUN SfT
    out = SfTJbbs(templateData.crsp.imgPts_mm,inputImageData.crsp.imgPts_n,options);
    phi = out.phi;
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%% EVALUATE THE METHOD
    % Use embedding phi to get correspondences in 3D
    output.inputImageData.crsp.camPts_mm = bbs_eval(phi.bbs,phi.ctrlpts,templateData.crsp.imgPts_mm(1,:)',templateData.crsp.imgPts_mm(2,:)',0,0);
else
    %% %%%%%%%%%%%%%%%%%%%%%%%%%% RUN SfT
    options.delta = computeDelta_SfT(templateData, options, 1);
    out = SfTJbbs(templateData.crsp.imgPts_mm,inputImageData.crsp.imgPts_n,options);
    phi = out.phi;
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%% EVALUATE THE METHOD
    % Use embedding phi to get correspondences in 3D
    output.inputImageData.crsp.camPts_mm = bbs_eval(phi.bbs,phi.ctrlpts,templateData.crsp.imgPts_mm(1,:)',templateData.crsp.imgPts_mm(2,:)',0,0);
end

output.phi = phi;
output.out = out;

%% %%%%%%%%%%%%%%%%%%%%%%%%%% DISPLAY
if WITHPLOT
    figure(5);
    clf;
    subplot(221);
    imshow(templateData.texturemap);
    hold on;
    plot(templateData.crsp.imgPts_px(1,:),templateData.crsp.imgPts_px(2,:),'c+');
    hold off;
    title('Correspondences on the texture-map');
    subplot(222);
    plot3(inputImageData.crsp.camPtsGT_mm(1,:),inputImageData.crsp.camPtsGT_mm(2,:),inputImageData.crsp.camPtsGT_mm(3,:),'r.');
    hold on;
    plot3(output.inputImageData.crsp.camPts_mm(1,:),output.inputImageData.crsp.camPts_mm(2,:),output.inputImageData.crsp.camPts_mm(3,:),'go');
    hold off;
    legend('GT crsp',['Estimated crsp using Chhatkuli17']);
    axis equal;
    title('Reconstruction of the correspondences')
end
