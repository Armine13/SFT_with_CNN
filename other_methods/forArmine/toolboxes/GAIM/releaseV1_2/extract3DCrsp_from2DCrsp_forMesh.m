function crspInputImage_mm = extract3DCrsp_from2DCrsp_forMesh(texturemap, inputImage, KK, meshInputImage, scale, crspTemplate_px)

numSamples = 10;
% Setup domain
uMin = 1;
uMax = size(texturemap,2);
vMin = 1;
vMax = size(texturemap,1);


evalPtsTemplate_px = crspTemplate_px;

% Get the 2D vertices in the texture-map
p_template = uv2PixCoords(meshInputImage.texMap.vertexUVW,size(texturemap,2),size(texturemap,1));
              
% We want to know the barycentric at the evaluation points in the
% texture-map
% Use the triangulation class of Matlab to do some processing
TR = triangulation(meshInputImage.faces,p_template);
% Compute the center of mass of each face
cMass = computeCenterMass(TR.ConnectivityList(:,:), p_template);

% Find the face where each evaluation point is lying on
PC = evalPtsTemplate_px';
ti = knnsearch(cMass,PC);

% Compute the barycentric coordinates [b1,b2,b3]
baryMask = cartesianToBarycentric(TR,ti,PC);
b1_evalPtsTemplate = baryMask(:,1)';
b2_evalPtsTemplate = baryMask(:,2)';
b3_evalPtsTemplate = baryMask(:,3)';

% Find the index of the three vertices associated to each face where each evaluation point is lying on
ind_v_evalPtsTemplate = meshInputImage.faces(ti,:)';

% Use the barycentric coordinates to get the position in 3D of each
% evaluation point
vertexPos3D = meshInputImage.vertexPos';
                
evalPtsInputImage_mm = repmat(b1_evalPtsTemplate,[3,1]).*vertexPos3D(:,ind_v_evalPtsTemplate(1,:)) + ...
                    repmat(b2_evalPtsTemplate,[3,1]).*vertexPos3D(:,ind_v_evalPtsTemplate(2,:)) + ...
                    repmat(b3_evalPtsTemplate,[3,1]).*vertexPos3D(:,ind_v_evalPtsTemplate(3,:));

% Project the 3D evaluation points to get the 2D evaluation points
% in the input image
evalPtsInputImage_n(1,:) = evalPtsInputImage_mm(1,:)./evalPtsInputImage_mm(3,:);
evalPtsInputImage_n(2,:) = evalPtsInputImage_mm(2,:)./evalPtsInputImage_mm(3,:);
evalPtsInputImage_n(3,:) = 1;
evalPtsInputImage_px = KK*evalPtsInputImage_n;

evalPtsTemplate_mm = evalPtsTemplate_px./scale;
evalPtsTemplate_mm(3,:) = 0;

% mm
% figure(21);
% clf;
% imshow(texturemap);
% hold on;
% plot(evalPtsTemplate_px(1,:),evalPtsTemplate_px(2,:),'c+');
% hold off;
% title('Texture-map with the evaluation points');             
% 
% figure(22);
% clf;
% plot3(evalPtsInputImage_mm(1,:),evalPtsInputImage_mm(2,:),evalPtsInputImage_mm(3,:),'r.');
% hold on;
% plot3(meshInputImage.vertexPos(:,1),meshInputImage.vertexPos(:,2),meshInputImage.vertexPos(:,3),'bo');
% hold off;
% axis equal;
% legend('Evaluation points in the deformed surface','3D GT correspondences');
% title('Checking the evaluation points in the deformed surface');
% 
% h = figure(23);
% clf;
% displayMatches2ImgswithColor(h,evalPtsTemplate_px(1:2,:),evalPtsInputImage_px(1:2,:),texturemap,inputImage);

crspInputImage_mm = evalPtsInputImage_mm;