% IPPE method for CVLAB dataset
clear all;
close all;
addpath(genpath('../ExtLibs/utilsmatlab'));
addpath('NLRefinement/');
addpath('BBS');
addpath('../ExtLibs/ris/');
addpath('../ExtLibs/cvx/sedumi/');
addpath('../ExtLibs/cvx/sdpt3/');
addpath('../ExtLibs/SALZ/');
load '../Salzseq/testDataSalzCVPR12';
%% Prepare template data
% Template image
img1=testData.mshRef.texMap.img;
% Calibration matrix
% K=testData.cam.K;
K = [528.0144   0.0000000e+00   3.2000000e+02; ...
   0.0000000e+00   528.0144   2.4000000e+02; ...
   0.0000000e+00   0.0000000e+00   1.0000000e+00];


focal=K(1,1);

% Get scale factor for reference image
vertexuv=testData.mshRef.texMap.vertexUVW(:,1:2);
vertex3D=testData.mshRef.vertexPos;
M=[vertex3D'-mean(vertex3D)'*ones(1,size(vertex3D,1))];
[U,D,V]=svd(M*M');
vertex3Dn=(V'*M)';
vertex=uv2PixCoords(vertexuv,size(img1,2),size(img1,1));
faces=testData.mshRef.faces;
% imshow(img1);
% hold on;
% plot(vertex(:,1),vertex(:,2),'r*');
% figure(2)
% hold on;
% plot3(vertex3D(:,1),vertex3D(:,2),vertex3D(:,3),'ro');
scale=norm(vertex3D(1,:)-vertex3D(size(vertex,1),:))./norm(vertex(1,:)-vertex(size(vertex,1),:));

for k = 20:20  %191
    %% Prepare input image and ground truth data
    Qxg=testData.corrData.corrs3D_GT{k}(1,:);
    Qyg=testData.corrData.corrs3D_GT{k}(2,:);
    Qzg=testData.corrData.corrs3D_GT{k}(3,:);
    P=find(Qzg~=0 & Qxg~=0 & Qyg~=0);
    % Ground Truth
    Pgth = [Qxg;Qyg;Qzg];
    Pgth = Pgth(:,P);  % ground-truth

    x2=testData.corrData.corrsImg{k}(1,:)';
    y2=testData.corrData.corrsImg{k}(2,:)';
    t1=testData.corrData.Brys{k}(:,1);
    alpha1=testData.corrData.Brys{k}(:,2);
    alpha2=testData.corrData.Brys{k}(:,3);
    alpha3=testData.corrData.Brys{k}(:,4);
    triangles=faces(t1,:);
    x1=alpha1.*vertex3Dn(triangles(:,1),1)+alpha2.*vertex3Dn(triangles(:,2),1)+alpha3.*vertex3Dn(triangles(:,3),1);
    y1=alpha1.*vertex3Dn(triangles(:,1),2)+alpha2.*vertex3Dn(triangles(:,2),2)+alpha3.*vertex3Dn(triangles(:,3),2);
    x2n=inv(K(1,1))*(x2-K(1,3));
    y2n=inv(K(2,2))*(y2-K(2,3));
    
    p=[x1';y1'];
    q=[x2n';y2n'];
    p = p(:,P);  % template
    q = q(:,P);  % image points in the camera frame
    
    
    %% CPBC15I method
    options.eta.er = 5e1;
    options.eta.nC = 12;
    options.phi.er = 5e2;
    options.phi.nC = 12;
    options.maxiter = 10;

    options.method = 'CPBC15I'; % Ajad PAMI
    outn = SfTJbbs(p,q,options);
    Qn = bbs_eval(outn.phi.bbs,outn.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
    
    %% BGCC12I method
    options.eta.er = 5e1;
    options.eta.nC = 12;
    options.phi.er = 5e2;
    options.phi.nC = 12;
    options.maxiter = 10;
    options.verbose = 0;
    options.method = 'BGCC12I'; % Adrien
    
    
    tic,
    outd = SfTJbbs(p,q,options);
    time_Bartoli12i = toc,
    
    
    %% MDH methods
    options.K = K;
    options.planar = true;
    options.outlierreject = 'none';
    options.method = 'Salz2';
    
    
    
    outs = SfTJbbs(p,q,options);  % Salzman
    
    
    
    options.method = 'Ostlund';
    outo = SfTJbbs(p,q,options);  % Ostlund

    Qs = bbs_eval(outs.phi.bbs,outs.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
    Qo = bbs_eval(outo.phi.bbs,outo.phi.ctrlpts,p(1,:)',p(2,:)',0,0);    
            
%    disp(sprintf('rmse Salzmann = %f',sqrt(mean(sum(Qs-Pgth).^2))));
%    disp(sprintf('rmse Ostlund = %f',sqrt(mean(sum(Qo-Pgth).^2))));
        
    options.eta.er = 5e2;
    options.phi.er = 5e0;
    options.method = 'CPB14I';
%     outj = SfTi(proi,qw,Jw,options);  


    tic,
    outj = SfTJbbs(p,q,options);   % Ajad CVPR
    time_Chhatkuli14 = toc    


    Qd = bbs_eval(outd.phi.bbs,outd.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
    Qj = bbs_eval(outj.phi.bbs,outj.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
%    disp(sprintf('rmse Adrien = %f',sqrt(mean(sum(Qd-Pgth).^2))));
%    disp(sprintf('rmse Ajad CVPR = %f',sqrt(mean(sum(Qj-Pgth).^2))));
%    disp(sprintf('rmse Ajad PAMI = %f',sqrt(mean(sum(Qn-Pgth).^2))));

%%
    % NL refinement of direct depth and stable methods
    options.phi.er = 1e0;
    options.lbdiso = 1e0;
    options.verbose = 1;

    tic,

    outdr = NLRefinebbs(outd.phi.bbs,outd.phi.ctrlpts,options,p,q,outd.proi,[],[]);    % Brunet

    Qdr = bbs_eval(outdr.phi.bbs,outdr.phi.ctrlpts,p(1,:)',p(2,:)',0,0);
   
    time_Brunet10 = toc


   %clc; 
   disp(sprintf('rmse Salzmann = %f',sqrt(mean(sum(Qs-Pgth).^2))));
   disp(sprintf('rmse Ostlund = %f',sqrt(mean(sum(Qo-Pgth).^2))));
   disp(sprintf('rmse Adrien = %f',sqrt(mean(sum(Qd-Pgth).^2))));
   disp(sprintf('rmse Ajad CVPR = %f',sqrt(mean(sum(Qj-Pgth).^2))));
   disp(sprintf('rmse Ajad PAMI = %f',sqrt(mean(sum(Qn-Pgth).^2))));
   disp(sprintf('rmse Brunet = %f',sqrt(mean(sum(Qdr-Pgth).^2))));            

   %%
    template = [ p/1000; zeros(1,size(p,2))];  % m
    xp = x2(P)'; % pixels
    yp = y2(P)'; % pixels
    object_shape3D = Pgth/1000;   % m

    input.template = template;
    input.pixels = [xp; yp];
    input.object_shape3D = object_shape3D;
    input.K = K;

    input.Salzman = sqrt(mean(sum(Qs-Pgth).^2));
    input.Ostlund = sqrt(mean(sum(Qo-Pgth).^2));
    input.Adrien = sqrt(mean(sum(Qd-Pgth).^2));
    input.AjadCVPR = sqrt(mean(sum(Qj-Pgth).^2));
    input.AjadPAMI = sqrt(mean(sum(Qn-Pgth).^2));
    input.Brunet = sqrt(mean(sum(Qdr-Pgth).^2));




    save(['input',num2str(k)],'input');
    k,
end


