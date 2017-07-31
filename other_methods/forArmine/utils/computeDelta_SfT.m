function delta = computeDelta_SfT(templateData, options, WITHPLOT)

% error('essayer d enlever KK');
er = options.delta.bbs.er;
nC = options.delta.bbs.nC;

uMin = min(templateData.mesh.texMap.vertexUVW(:,1));
uMax = max(templateData.mesh.texMap.vertexUVW(:,1));
vMin = min(templateData.mesh.texMap.vertexUVW(:,2));
vMax = max(templateData.mesh.texMap.vertexUVW(:,2));

%% Prepare the embedding
delta.bbs = bbs_create(uMin, uMax, nC, vMin, vMax, nC, 3);

coloc = bbs_coloc(delta.bbs, templateData.mesh.texMap.vertexUVW(:,1)', templateData.mesh.texMap.vertexUVW(:,2)');
lambdas = er*ones(nC-3, nC-3);
bending = bbs_bending(delta.bbs, lambdas);

cpts = (coloc'*coloc + bending) \ (coloc'*templateData.mesh.vertexPos(:,1:3));
delta.ctrlpts = cpts';

Qest = (coloc*delta.ctrlpts')';

qest_x = Qest(1,:)./Qest(3,:);
qest_y = Qest(2,:)./Qest(3,:);

qest_n = [qest_x;qest_y;ones(1,length(qest_x))];

qest = templateData.KK*qest_n;
qest = qest(1:2,:);

%% Check the error
error_2D = sqrt(mean((qest(1,:)-templateData.mesh.texMap.vertexUVW(:,1')).^2+(qest(2,:)-templateData.mesh.texMap.vertexUVW(:,2)').^2));
error_3D = sqrt(mean((Qest(1,:)-templateData.mesh.vertexPos(:,1)').^2+(Qest(2,:)-templateData.mesh.vertexPos(:,2)').^2+(Qest(3,:)-templateData.mesh.vertexPos(:,3)').^2));

disp(['Delta function estimation: [2D Prediction Error = ' num2str(error_2D) ' px][3D Prediction Error = ' num2str(error_3D) ' mm]' ]);

%% Display
if WITHPLOT
    figure(10);
    clf;
    subplot(121);
    plot(templateData.mesh.texMap.vertexUVW(:,1), templateData.mesh.texMap.vertexUVW(:,2),'b+');
    hold on;
    plot(qest(1,:), qest(2,:),'b+');
    hold off;
    legend('GT points','Estimated points');
    title('Reprojection of points computed using the Delta function');
    subplot(122);
    plot3(templateData.mesh.vertexPos(:,1), templateData.mesh.vertexPos(:,2), templateData.mesh.vertexPos(:,3),'b+');
    hold on;
    plot3(Qest(1,:), Qest(2,:), Qest(3,:), 'b+');
    hold off;
    legend('GT points','Estimated points');
    title('3D points computed using the Delta function');
end