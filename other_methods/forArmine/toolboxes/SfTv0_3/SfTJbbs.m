% ver 0.3: Replacing TPS with BBS warps
%          New SfT method added
% Robust estimation of scale
% Addition of ground truth for scale
% Change iterative methods to use gt scale
% Shape from Template
%
% SYNTAX
%   [out]=SfT(p,q,options)
%  
% INPUT ARGUMENTS
%   -p: (2xn) array of n 2D points in the flat template
%   -q: (2xn) array of b 2D points in the image. q must be normalised with
%   the intrinsic parameter matrix of the camera.
%   -options: structure with the following fields
%           'eta': struct with eta (template to image warp) parameters%                  
%                  eta.er: smoothing. default=5e2;
%                  eta.nC: nC^2 control centers. default=20;
%           'phi': struct with phi (template to shape warp) parameters
%                   phi.er: smoothing. default=0.55;
%                   phi.nC: nC^2 control centers. default=20;
%           'verbose': 1 --> gives intermediate results
%           'KLims': Rectangle bounds of the template.
%                     KLims=[umin,umax,vmin,vmax]
%           'method': method used for shape estimation
%                    'BGCC12I' --> Analytical solution for Isometric
%                    Deformations.
%                    'BGCC12IR' --> Analytical solution + NL Refinement for
%                    Isometric Deformations
%                    'BGCC12C' --> Analytical solution for Conformal
%                    Deformations
%                    'BGCC12CR' --> Analytical solution + NL Refinement for 
%                    Conformal Deformations
%                    'CPB14I' --> Analytical solution for Isometric
%                    Deformations with the solution of Jacobian
%                    'CPB14IR' --> NL Refinement method
%                    initialized with 'CPB14I' solution.
%                    'CPBC15I' --> (New) Analytical solution for Isometric
%                    deformations based on the solution of normals.
%           'NGridx': number of grid points in x used to sample the template (default=50)
%           'NGridy': number of grid points in y used to sample the template (default=50).
%           'maxiter': number of max iterations in case of refinement
%           (default=40)
%           'delta': (Warning: delta must be provided only if the template is 3D !) 
%            warp from flat template to 3D template. Structure with TPS parameters   
%                  delta.er: external smoothing
%                  delta.nC: number of control centers
%                  delta.bbs: bbs structure of Brunet's toolbox
%                  delta.ctrlpts: bbs control points
% OUTPUT ARGUMENTS
%       out: output structure with the following fields:
%           'phi': solution to shape. structure with TPS parameters   %                  
%                  phi.bbs: bbs structure of Brunet's toolbox
%                  phi.ctrlpts: bbs control points
% IMPORTANT INTERMEDIATE VARIABLES
%   'Jthetaprime': Differentiation of theta
%        'Jtheta': Direct solution for Jacobian with Type I PDE

%   This code is partially based on the work of  
%   [Bartoli et. al 2012] On Template-Based Reconstruction from a Single View:
%   Analytical Solutions and Proofs of Well-Posedness for  Developable,
%   Isometric and Conformal Surfaces
%   Method 'CPB's are partially based on [Chhatkuli et. al 2014]
%   Stable Template-Based Isometric 3D Reconstruction
%   in All Imaging Conditions by Linear Least-Squares.
%   Non Linear Refinement methods are based on [Brunet et. al 2014]
%   Monocular template-based 3D surface reconstruction: Convex inextensible
%   and nonconvex isometric methods
%   Method 'CPBC's are partially based on [Chhatkuli et. al 2015] submitted
%   to TPAMI
%   (c) 2013, Adrien Bartoli and Daniel Pizarro. dani.pizarro@gmail.com

%
% Sft is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% Sft is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
% for more details.
% 
% You should have received a copy of the GNU General Public License along
% with this program; if not, write to the Free Software Foundation, Inc.,
% 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

% SfT Shape from Template source code
function [out]=SfTJbbs(p,q,options)
if(nargin<2)
    disp('Error: 2xn arrays p1 and p2 are needed...');
    coeff=[];
    out=[];
    return
elseif(nargin==2)
    [options,error]=ProcessArgs(p,q);
else
    [options,error]=ProcessArgs(p,q,options);
end

er = options.eta.er;
nC = options.eta.nC;
umin = options.KLims(1); umax = options.KLims(2);
vmin = options.KLims(3); vmax = options.KLims(4);
bbs = bbs_create(umin, umax, nC, vmin, vmax, nC, 2);

coloc = bbs_coloc(bbs, p(1,:), p(2,:));
lambdas = er*ones(nC-3, nC-3);
bending = bbs_bending(bbs, lambdas);

% get control points for i to j warp
cpts = (coloc'*coloc + bending) \ (coloc'*q(1:2,:)');
ctrlpts = cpts';

qw = bbs_eval(bbs,ctrlpts,p(1,:)',p(2,:)',0,0);
% % Get Warp Centers
% C = TPSGenerateCenters(options.eta.nC,options.KLims+1e-3.*[-1,1,-1,1]);
% % Precompute EpsilonLambda matrix
% EpsilonLambda=TPSEpsilonLambda(C,options.eta.ir);
% % Get warp parameters from features
% L = TPSWfromfeatures(p,q,C,options.eta.er,options.eta.ir,EpsilonLambda);
% % Warp points p and get warp reprojection error
% [~,qw]=TPSWarpDiff(p,L,C,options.eta.ir,EpsilonLambda);
error=sqrt(mean((qw(1,:)-q(1,:)).^2+(qw(2,:)-q(2,:)).^2));
if(options.verbose)
    %Visualize Point Registration Error
    [xv,yv]=meshgrid(linspace(options.KLims(1),options.KLims(2),20),linspace(options.KLims(3),options.KLims(4),20));
    
    qv = bbs_eval(bbs,ctrlpts,xv(:),yv(:),0,0);
    
    disp([sprintf('[ETA] Internal Rep error = %f',error)]);
    figure;
    plot(q(1,:),q(2,:),'ro');
    hold on;
    plot(qw(1,:),qw(2,:),'b*');
    mesh(reshape(qv(1,:),size(xv)),reshape(qv(2,:),size(xv)),zeros(size(xv)));    
    hold off;    
end
out.eta.p=p;
out.eta.q=q;
out.eta.bbs=bbs;
out.eta.ctrlpts=ctrlpts;
out.eta.er=options.eta.er;

switch(options.method)
    case{'BGCC12I','BGCC12IR'} % Analytical Direct solution for isometry and perspective camera
        if(~(isfield(options.phi,'ctrlpts') && strcmp(upper(options.method),'BGCC12IR'))) % Is user giving an initialization of phi ?        
        % Create a grid of points to go to 3D
        NRoi=options.NGridy*options.NGridx;
        [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
        proi=[xroi(:)';yroi(:)'];
        % Get Derivatives
        dqu = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',1,0);
        dqv = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,1);
        dq = [dqu;dqv];
        qw = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,0);
        
        % Get phi points
        gamma=zeros(1,NRoi);
        Q=zeros(3,NRoi);
        if(isfield(options,'delta'))
            delta = options.delta;
            dpu = bbs_eval(delta.bbs, delta.ctrlpts, proi(1,:)',proi(2,:)',1,0);
            dpv = bbs_eval(delta.bbs, delta.ctrlpts, proi(1,:)',proi(2,:)',0,1);
            dp = [dpu;dpv];
        end
        for i=1:NRoi
            Jdelta=eye(2);
            if(isfield(options,'delta'))
                Jdelta=[dp(1:3,i) dp(4:6,i)];
            end
            eta=[qw(1,i);qw(2,i)];
            Jeta=[dq(1,i),dq(3,i);dq(2,i),dq(4,i)];
             %A=(inv(V)*((Jeta'*(eye(2)-(eta*eta')./(eta'*eta+1))*Jeta))*inv(V'))./(eta'*eta+1);
            M=(Jeta'*(eye(2)-(eta*eta')./(eta'*eta+1))*Jeta)/(Jdelta'*Jdelta);
            eigM=svd(M);
            gamma(i)=1./(sqrt(max(eigM)));
            Q(1,i)=gamma(i)*eta(1);
            Q(2,i)=gamma(i)*eta(2);
            Q(3,i)=gamma(i);
        end        
        
        % Get Warp Centers
        % 2D to 3D warp
        nC = options.phi.nC;
        er = options.phi.er;
        
        bbs3 = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);

        coloc3 = bbs_coloc(bbs3, proi(1,:), proi(2,:));
        lambdas3 = er*ones(nC-3, nC-3);
        bending3 = bbs_bending(bbs3, lambdas3);

        % get control points for p to Q warp
        cpts3 = (coloc3'*coloc3 + bending3) \ (coloc3'*Q(1:3,:)');
        ctrlpts3 = cpts3';

        out.phi.Q=Q;
        out.proi=proi;
        out.phi.bbs = bbs3;
        out.phi.ctrlpts = ctrlpts3;
        out.phi.er=er;
        out.phi.nC=options.phi.nC;
        else
          out.phi=options.phi;
        end
        switch(options.method)
            case 'BGCC12IR'
                out = NLrefine_ris(out.phi.bbs, out.phi.ctrlpts, p, q, ...
                      options.rephi.iso, options.rephi.bend);                                
        end    
 
    case {'CPB14I','CPB14IR'} % Analytical solution from the Jacobian for isometry and perspective camera
        if(~(isfield(options.phi,'ctrlpts') && strcmp(lower(options.method),'CPB14IR'))) % Is user giving an initialization of phi ?        
        % Create a grid of points to go to 3D
        NRoi=options.NGridy*options.NGridx;
        [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
        proi=[xroi(:)';yroi(:)'];
        % Get Derivatives of eta (Jeta)
        dqu = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',1,0);
        dqv = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,1);
        dq = [dqu; dqv];
        qw = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,0);
        
        % Get phi points
        gamma=zeros(1,NRoi); % Depth variables
        Q=zeros(3,NRoi); % 3D Points from the solution of the Jacobian
        Jtheta = zeros(2,NRoi); % Jacobian of theta from direct solution
        theta = zeros(1,NRoi); % theta from direct solution
        Epsilon = zeros(1,NRoi);
        Gmat = zeros(2,2,NRoi); % Matrix xi for all points
        
        % Get derivatives of Delta if Delta is given
        if(isfield(options,'delta'))
            delta = options.delta;
            dpu = bbs_eval(delta.bbs, delta.ctrlpts, proi(1,:)',proi(2,:)',1,0);
            dpv = bbs_eval(delta.bbs, delta.ctrlpts, proi(1,:)',proi(2,:)',0,1);
            dp = [dpu;dpv];
        end
        n = zeros(3,NRoi);  % analytic normals
       
            for i=1:NRoi
                Jdelta=eye(2);
                % if Delta given, use the above computed values
                if(isfield(options,'delta'))
                    Jdelta=[dp(1:3,i) dp(4:6,i)];
                end
                % Use eta and computed derivatives of eta
                eta=[qw(1,i);qw(2,i)];
                Jeta=[dq(1,i),dq(3,i);dq(2,i),dq(4,i)];
                
                % Get Cholesky decomposition of G
                epsilsq = eta'*eta+1; % squared norm of eta
                G = 1/epsilsq * (Jeta'*(eye(2)-(eta*eta')./epsilsq)*Jeta);
                Gmat(:,:,i) = G;
                V = (chol(G))'; % Lower triangular matrix of Cholesky decomposition
                
                % Get eigenvalues and eigen matrix
                Ae=inv(V)*(Jdelta'*Jdelta)*inv(V');
                [eigvA,eigA,~]=svd(Ae);
                eigvA=eigvA*sign(eigvA(1,1));

                % Get maximum eigenvalue index
                if eigA(1,1)>eigA(2,2)
                    indmax = 1;
                    indmin = 2;
                else
                    indmax = 2;
                    indmin = 1;
                end

                Epsilon(i) = sqrt(epsilsq);
%                 indmin = 3-indmax; % minimum eigenvalue index                                
                theta(i)=(sqrt(eigA(indmin,indmin))); % Square root of the second eigenvalue
                gamma(i) = theta(i)/Epsilon(i);                
                Jt = sqrt(eigA(indmax,indmax)-eigA(indmin,indmin))*V*eigvA(:,indmax); 
                Jtheta(:,i) = Jt;
                
                n(:,i) = analyticNormals(eta,Jeta,Jt,theta(i),epsilsq);
                
                Q(1,i)=gamma(i)*eta(1);
                Q(2,i)=gamma(i)*eta(2);
                Q(3,i)=gamma(i);                
            end
            
            % theta warp options:
            nC = options.phi.nC;
            er = options.phi.er;
            % Compute BBS Warp of theta obtained from direct computation
            % Warp: R^2-->R^2
            bbs1 = bbs_create(umin, umax, nC, vmin, vmax, nC, 1);

            coloc1 = bbs_coloc(bbs1, proi(1,:), proi(2,:));
            lambdas1 = er*ones(nC-3, nC-3);
            bending1 = bbs_bending(bbs1, lambdas1);

            % get control points for i to j warp
            cpts1 = (coloc1'*coloc1 + bending1) \ (coloc1'*theta');
            ctrlpts1 = cpts1';
%             Jthetaprime = bbs_eval(bbs1,ctrlpts1,proi(1,:)',proi(2,:)',1,1);
            Jthetau = bbs_eval(bbs1,ctrlpts1,proi(1,:)',proi(2,:)',1,0);
            Jthetav = bbs_eval(bbs1,ctrlpts1,proi(1,:)',proi(2,:)',0,1);
            Jthetaprime = [Jthetau; Jthetav];
            
            thetaprime = bbs_eval(bbs1,ctrlpts1,proi(1,:)',proi(2,:)',0,0);
                                                
            Jthetaprimenorm = sqrt(Jthetaprime(1,:).^2+Jthetaprime(2,:).^2);
            Jthetanorm = sqrt(Jtheta(1,:).^2 + Jtheta(2,:).^2);
            
            Jang = (Jthetaprime./[Jthetaprimenorm; Jthetaprimenorm]).*(Jtheta./[Jthetanorm; Jthetanorm]);            
            Jdot = abs(sum(Jang));
            
            medAng = median(Jdot);
            th = medAng - 3.0*std(Jdot); % threshold for removing points
            NRoi_r = NRoi; % Actual number of points after removing bad derivatives.
            proi_r = proi;
            indices = 1: length(proi_r); % Indices of preserved points
            Q_rj = Q;
            Jthetaj = Jtheta;
            Jthetaprimej = Jthetaprime;
            thetaprimej = thetaprime;
            rem = []; % removal indices            
            for i = 1: NRoi
                if Jdot(i) < th
                    NRoi_r = NRoi_r -1;
                    rem = [rem;i]; % removal index
                end
            end
            proi_r(:,rem) = [];
            indices(rem) = [];
            Q_rj(:,rem) = [];
            thetaprimej(:,rem) = [];
            Jthetaj(:,rem) = []; % Test 2
            Jthetaprimej(:,rem) = []; % Test 2
            
% ************************** Integrate Jthetaj to obtain theta ************
            % Carry sign from TPS derivative
            Jthetaj = flipSigns(Jthetaj,Jthetaprimej);
            
            ctrlptsint = BBSIntegration(bbs1,proi_r,Jthetaj,er/60000);
            thetapj = bbs_eval(bbs1,ctrlptsint,proi_r(1,:)',proi_r(2,:)',0,0);
                        
            scaleI = median(thetaprimej-thetapj);
            thetahatj = thetapj + scaleI;
            
            for i = 1:NRoi_r
                eta=[qw(1,indices(i));qw(2,indices(i))];
                gammaj = thetahatj(i)/Epsilon(indices(i));                
                Q_rj(1,i)=gammaj*eta(1);
                Q_rj(2,i)=gammaj*eta(2);
                Q_rj(3,i)=gammaj;
            end
            % Original direct method
            bbs3 = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);

            coloc3 = bbs_coloc(bbs3, proi(1,:)', proi(2,:)');
            lambdas3 = er*ones(nC-3, nC-3);
            bending3 = bbs_bending(bbs3, lambdas3);
            ctrlpts3 = (coloc3'*coloc3 + bending3) \ (coloc3'*Q(1:3,:)');
            ctrlpts3 = ctrlpts3';

            % Removal of points in the Jacobian method
            coloc3rj = bbs_coloc(bbs3, proi_r(1,:)', proi_r(2,:)');
            ctrlpts3rj = (coloc3rj'*coloc3rj + bending3) \ (coloc3rj'*Q_rj(1:3,:)');
            ctrlpts3rj = ctrlpts3rj';

            % Original method
            Qw = bbs_eval(bbs3,ctrlpts3,proi(1,:)',proi(2,:)',0,0);

            % From integration of Jthetaj by removing points with bad
            % derivatives
            Qpj = bbs_eval(bbs3,ctrlpts3rj,proi(1,:)',proi(2,:)',0,0);

            out.phi.Q=Qpj;
        
            out.phi.ctrlpts=ctrlpts3rj;
            out.phi.bbs=bbs3;
            out.phi.er=options.phi.er;
            out.proi = proi;
        else
          out.phi=options.phi;
        end
        switch(options.method)
            case 'CPB14IR'
                out = NLrefine_ris(options.phi.bbs, options.phi.ctrlpts, p, q, ...
                      options.rephi.iso, options.rephi.bend);                                
        end
        
    case {'CPBC15I','CPB15IR','CPBC15IR'}
        if(~(isfield(options.phi,'ctrlpts') && strcmp(upper(options.method),'CPBC15I'))) % Is user providing with an initialization of phi ?
            NRoi=options.NGridy*options.NGridx;
            [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
            proi=[xroi(:)';yroi(:)'];
            % Get Derivatives of eta (Jeta)
            dqu = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',1,0);
            dqv = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,1);
            dq = [dqu; dqv];
            qw = bbs_eval(bbs, ctrlpts, proi(1,:)',proi(2,:)',0,0);
            
            Jw = [dq(1,:);dq(3,:);dq(2,:);dq(4,:)];            
            [Q, N1, N2] = IPPEO(qw, Jw);

%             [N1o,N2o] = IPPEwp(Jw);
%             Qwpn = Shapefrom2Normals(N1o,N2o,p,proi,qw,Q,options.phi.nC,options.phi.er);

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

            % Find a single normal field
            N = disambiguate_normals(N1,N2,Nd);

            % Do shape from normals to get the embedding
            pt = [qw; ones(1,size(qw,2))];
            nC=28;
            lambdas = 10e-3*ones(nC-3, nC-3); %10e-3
            bbsd = bbs_create(umin, umax, nC, vmin, vmax, nC, 1);
            colocd = bbs_coloc(bbsd, pt(1,:), pt(2,:));
            bendingd = bbs_bending(bbsd, lambdas);
            [ctrlpts3Dn]=ShapeFromNormals(bbsd,colocd,bendingd,pt,N);
            mu=bbs_eval(bbsd, ctrlpts3Dn, pt(1,:)', pt(2,:)',0,0);
            % Correct integration constant:
            % mu = mu *(median(Qw(3,:))/median(mu));
            Qd = [qw(1,:).*mu; qw(2,:).*mu; mu];

            [bbs3,ctrlpts3] = make_bbs_warp(proi,Qd,options.phi.nC,options.phi.er,options.KLims);
%             Qn = bbs_eval(bbs3,ctrlpts3,p(1,:)',p(2,:)',0,0);
            Qn = RegisterToGTH(Qd,Q);
            
            [bbs3r,ctrlpts3r] = make_bbs_warp(proi,Qn,options.phi.nC,options.phi.er,options.KLims);                        
            out.phi.Q = Qn;
            out.phi.bbs = bbs3r;
            out.phi.ctrlpts = ctrlpts3r; 
            out.proi = proi;
        else
            out.phi=options.phi;
        end
        switch(options.method)
            case 'CPBC15IR'
                out = NLrefine_ris(out.phi.bbs, out.phi.ctrlpts, p, q, ...
                    options.rephi.iso, options.rephi.bend);
        end           

   
    case 'PeIso'        
        corTerm=1;
        %[depth,linkTo,Q,linkAngle]=inextensibleTemplateBounding(p,q,[eye(3),zeros(3,1)],1);
        %depth=real(depth);
        [depth,linkTo,Q,linkAngle] = ris_perriollat_bounding(p,q,[eye(3),zeros(3,1)],1);
        %[depth,Q] = surfaceOptimisation(p,q,[eye(3),zeros(3,1)],depth,linkAngle,linkTo,corTerm);
         % Get Warp Centers
        Q=real(Q);
        
        er = options.phi.er;
        nC = options.phi.nC;
        umin = options.KLims(1); umax = options.KLims(2);
        vmin = options.KLims(3); vmax = options.KLims(4);
        bbs3 = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);

        coloc = bbs_coloc(bbs3, p(1,:), p(2,:));
        lambdas = er*ones(nC-3, nC-3);
        bending = bbs_bending(bbs3, lambdas);

        % get control points for i to j warp
        cpts = (coloc'*coloc + bending) \ (coloc'*Q(1:3,:)');
        ctrlpts3 = cpts';
        
        % Create a grid of points to go to 3D
        NRoi=options.NGridy*options.NGridx;
        [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
        proi=[xroi(:)';yroi(:)'];
        
        Qw = bbs_eval(bbs3,ctrlpts3,proi(1,:)',proi(2,:)',0,0);        
%         error=sqrt(mean((Qw(1,:)-Q(1,:)).^2+(Qw(2,:)-Q(2,:)).^2+(Qw(3,:)-Q(3,:)).^2));
%         if(options.verbose)
%             %Visualize Point Registration Error
%             disp([sprintf('[PHI] Internal Rep error = %f',error)]);
%         end
        out.phi.Q=Q;
        out.phi.p=proi;
        out.phi.bbs = bbs3;
        out.phi.ctrlpts = ctrlpts3;        
        out.phi.er=options.phi.er;
        
    case 'BrIso'
        Q = ris_richard_errvect(p, q, [eye(3),zeros(3,1)], 0.4, 1e-3);
         % Get Warp Centers
        
        er = options.phi.er;
        nC = options.phi.nC;
        umin = options.KLims(1); umax = options.KLims(2);
        vmin = options.KLims(3); vmax = options.KLims(4);
        bbs3 = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);

        coloc = bbs_coloc(bbs3, p(1,:), p(2,:));
        lambdas = er*ones(nC-3, nC-3);
        bending = bbs_bending(bbs3, lambdas);

        % get control points for i to j warp
        cpts = (coloc'*coloc + bending) \ (coloc'*Q(1:3,:)');
        ctrlpts3 = cpts';
        
        % Create a grid of points to go to 3D
        NRoi=options.NGridy*options.NGridx;
        [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
        proi=[xroi(:)';yroi(:)'];
        
        Qw = bbs_eval(bbs3,ctrlpts3,proi(1,:)',proi(2,:)',0,0);        
%         error=sqrt(mean((Qw(1,:)-Q(1,:)).^2+(Qw(2,:)-Q(2,:)).^2+(Qw(3,:)-Q(3,:)).^2));
%         if(options.verbose)
%             %Visualize Point Registration Error
%             disp([sprintf('[PHI] Internal Rep error = %f',error)]);
%         end
        out.phi.Q=Q;
        out.phi.p=proi;
        out.phi.bbs = bbs3;
        out.phi.ctrlpts = ctrlpts3;        
        out.phi.er=options.phi.er;
        
    case 'Salz2'        
        % Initialization for the Salzmann's method
        n_corresp = size(p,2);
        npts_mesh = 20;
        [tri tri_x tri_y tri_z] = ris_create_tri_mesh(options.KLims(1), options.KLims(2), npts_mesh, options.KLims(3), options.KLims(4), npts_mesh);
        
        % Convert the template points into barycentric coordinates
        [b1 b2 b3 ind_tri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z,p(1,:),p(2,:), ones(size(p(2,:))));
        
        meshIn.vertexPos=[tri_x,tri_y,tri_z];
        meshIn.faces=tri;
        K = options.K;
%         K=[500,0,320;0,500,250;0,0,1];
        qn=[q;ones(1,size(q,2))];
        qn=K*qn;
        
        
        tic,
        
        [meshOut,Q]=SalzReconstruction(qn(1:2,:),ind_tri,[b1;b2;b3]',meshIn,K);
        %[xx yy zz] = ris_bary_to_cart(tri, x, y, z, ind_tri, b1, b2, b3);
        %Q = [xx ; yy ; zz];
         
        time_Salzman11 = toc
        
        
        er = options.phi.er;
        nC = options.phi.nC;
        umin = options.KLims(1); umax = options.KLims(2);
        vmin = options.KLims(3); vmax = options.KLims(4);
        bbs3 = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);

        coloc = bbs_coloc(bbs3, p(1,:), p(2,:));
        lambdas = er*ones(nC-3, nC-3);
        bending = bbs_bending(bbs3, lambdas);

        % get control points for i to j warp
        cpts = (coloc'*coloc + bending) \ (coloc'*Q(1:3,:)');
        ctrlpts3 = cpts';    
        
        % Create a grid of points to go to 3D
        NRoi=options.NGridy*options.NGridx;
        [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
        proi=[xroi(:)';yroi(:)'];
        
        out.phi.Q=Q;
        out.phi.p=proi;
        out.phi.bbs = bbs3;
        out.phi.ctrlpts = ctrlpts3;        
        out.phi.er=options.phi.er;
                
    case 'Ostlund'
        % Initialization for Ostlund's method
        npts_mesh = 20;
        [tri tri_x tri_y tri_z] = ris_create_tri_mesh(options.KLims(1), options.KLims(2), npts_mesh, options.KLims(3), options.KLims(4), npts_mesh);        
        
        % Convert the template points into barycentric coordinates
        [b1 b2 b3 ind_tri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z, p(1,:), p(2,:), ones(size(p(2,:))));        
        
        meshIn.vertexPos=[tri_x,tri_y,tri_z];
        meshIn.faces=tri;

        
        tic,
        
        % Sample control vertices using Fast marching [Peyre]
        [ctrlinds] = furthestPointSampler(meshIn.vertexPos',30,[]);
        
        K = options.K;
%         K=[500,0,320;0,500,240;0,0,1];
        
        qn=[q;ones(1,size(q,2))];
        qn=K*qn;
%         [i1 i2 i3 ind_imtri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z,qn(1,:),qn(2,:),ones(size(qn(2,:))));
               
%         U = qn';
%         bw = [b1;b2;b3];
%         bi = tri(ind_tri,:);
%         pts = zeros(3,length(U));
%         dim = length(U);
%         for i = 1:3,
%             pts = pts + bw(:, i*ones(1, dim)).*vertices(bi(:, i),:);
%         end        
        init_Ost;
        [Q,meshOut] = OstReconstructionPlanar(qn(1:2,:),ind_tri,[b1;b2;b3]',meshIn,K,ctrlinds,options);
        
        Qc = [b1;b1;b1].*Q(:,tri(ind_tri,1))+[b2;b2;b2].*Q(:,tri(ind_tri,2))+[b3;b3;b3].*Q(:,tri(ind_tri,3));
        
        
        
%         time_Ostlund12 = toc
%         
%         [meshOut,Q]=SalzReconstruction(qn(1:2,:),ind_tri,[b1;b2;b3]',meshIn,K);
        %[xx yy zz] = ris_bary_to_cart(tri, x, y, z, ind_tri, b1, b2, b3);
        %Q = [xx ; yy ; zz];        
        
        er = options.phi.er;
        nC = options.phi.nC;
        umin = options.KLims(1); umax = options.KLims(2);
        vmin = options.KLims(3); vmax = options.KLims(4);
        bbs3 = bbs_create(umin, umax, nC, vmin, vmax, nC, 3);

        coloc = bbs_coloc(bbs3, p(1,:), p(2,:));
        lambdas = er*ones(nC-3, nC-3);
        bending = bbs_bending(bbs3, lambdas);

        % get control points for i to j warp
        cpts = (coloc'*coloc + bending) \ (coloc'*Qc(1:3,:)');
        ctrlpts3 = cpts';

        
        % Create a grid of points to go to 3D
        NRoi=options.NGridy*options.NGridx;
        [xroi,yroi]=meshgrid(linspace(options.KLims(1),options.KLims(2),options.NGridx),linspace(options.KLims(3),options.KLims(4),options.NGridy));
        proi=[xroi(:)';yroi(:)'];
        
        Qw = bbs_eval(bbs3,ctrlpts3,proi(1,:)',proi(2,:)',0,0);
        
%         error=sqrt(mean((Qw(1,:)-Qc(1,:)).^2+(Qw(2,:)-Qc(2,:)).^2+(Qw(3,:)-Qc(3,:)).^2));
%         if(options.verbose)
%             %Visualize Point Registration Error
%             disp([sprintf('[PHI] Internal Rep error = %f',error)]);
%         end
        out.phi.Q=Qc;
        out.phi.p=proi;
        out.phi.bbs = bbs3;
        out.phi.ctrlpts = ctrlpts3;        
        out.phi.er=options.phi.er;
end

end

function [options,error]=ProcessArgs(p1,p2,options)
% hard coded defaults
nC=20;er=1e-3;
%
error=[];
[d,n]=size(p1);
[d2,n2]=size(p2);
if(d<2 || d2<2 || n2~=n || n<3)
    error='Point arrays with mismatched dimmensions or few points given...';
    options=[];
    return
end
if(nargin<3)
options=[];
end
if(~isfield(options,'eta'))
        options.eta.er=er;
        options.eta.nC=nC;
else
    if(~isfield(options.eta,'er'))
        options.eta.er=er;        
    end

    if(~isfield(options.eta,'nC'))
        options.eta.nC=nC;        
    end
end

if(~isfield(options,'phi'))
        options.phi.er=er;
        options.phi.nC=nC;        
else
    if(~isfield(options.phi,'er'))
        options.phi.er=er;        
    end
    if(~isfield(options.phi,'nC'))
        options.phi.nC=nC;        
    end
end

if(~isfield(options,'rephi'))
    options.rephi.iso = 50;
    options.rephi.bend = 200;
else    
    if(~isfield(options.rephi,'iso'))
        options.rephi.iso = 50;
    end

    if(~isfield(options.rephi,'bend'))
        options.rephi.bend = 200;
    end    
end

if(~isfield(options,'verbose'))
        options.verbose=0;
end

if(~isfield(options,'KLims'))
    umin=min(p1(1,:));
    umax=max(p1(1,:));
    vmin=min(p1(2,:));
    vmax=max(p1(2,:));
    options.KLims=[umin,umax,vmin,vmax];
end
if(~isfield(options,'method'))
    options.method='AnJ';
end
if(~isfield(options,'NGridx'))
    options.NGridx=50;
end
if(~isfield(options,'NGridy'))
    options.NGridy=50;
end

end

function [ Jtheta ] = flipSigns( Jtheta, Jthetaprime )
    for i = 1: size(Jtheta,2)
        if abs(Jtheta(1,i))> abs(Jtheta(2,i))
            if sign(Jtheta(1,i))~=sign(Jthetaprime(1,i))
                Jtheta(:,i) = - Jtheta(:,i);
            end
        else
            if sign(Jtheta(2,i))~=sign(Jthetaprime(2,i))
                Jtheta(:,i) = - Jtheta(:,i);
            end
        end
    end
end

function ni = analyticNormals(eta,Jeta,Jt,thetai,epsilsq)

etah = [eta;1];
Jetah = [Jeta; 0 0];
epsilon = sqrt(epsilsq);
Jeps = 1/epsilon*eta'*Jeta;

% Computation of Jphi
Jphi = 1/epsilon*(etah*Jt') -thetai/epsilsq*etah*Jeps +thetai/epsilon*Jetah;

% Normal computation
n = -cross(Jphi(:,1),Jphi(:,2));
ni = n/norm(n);

end