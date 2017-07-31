function [meshOut,P]=SalzReconstructionL(p,v,b,mesh,A,ncx,ncy)
% p 2xN points detected in the image
% v face index for each point p in the template
% b barycentric coordinates of the points p nx3 matrix
%mesh.faces ---->Faces of the mesh
%mesh.B     ----> Barycentrics of the mesh
%mesh.vertexPos ----> Vertexes of the mesh
% A is the intrinsic calibration matrix
% meshOut is the triangulated mesh

Rt=[eye(3),zeros(3,1)];
tri=mesh.faces-1;
[patches,wvs] = make_patches(0,0,5,5,ncx,ncy,1);
CwR=mesh.vertexPos;
E = compute_edges2(tri,size(CwR,1));
L = zeros(size(E,1),1);
for i=1:length(L)
    t = CwR(E(i,2),:) - CwR(E(i,1),:);
    L(i) = sqrt(t*t');
end
nC = size(CwR,1);

wri = 10;
if(wri > 0)    
    Var = load('./toolboxes/B_ReconstructionMethods/Salzmann09_11/data/modes.var');
    PC = load('./toolboxes/B_ReconstructionMethods/Salzmann09_11/data/modes.mds');
    PC = PC';
    PCc = transform_modes_camera(PC,Rt);
    nM = 75;
    nP = size(patches,1);
    Cw0= CwR;
    tmp = [Cw0';ones(1,nC)];
    CcR = reshape(Rt*tmp,3*nC,1);
    [P,Vr] = compute_local_pca_matrix(patches,wvs,nM,PCc,Var,CcR);
    [Al,cl,ql] = socpLocalPrior(P,Vr,nC);
end


pars=[];
pars.fid=1;

% Precompute socp length constraints
[Ac,cc,qc] = socpConstraints([E,L],nC);
 

%
wr = wri;
matches_all=[mesh.faces(v,:)-1,b,p'];
    if(wr > 0)
        Nr = compute_patches_matches(matches_all,patches);
        Wr = wr*speye(nM*size(Nr,2));
    end

% Compute homogeneus linear system for point reprojection
M = compute_correspondence_matrix(A,b,mesh.faces(v,:)-1,p',1,nC);    
uvs = [p;ones(1,size(p,2))];
S = inv(A)*uvs;
C = sparse(getVToXMatrix(b,mesh.faces(v,:)-1,nC));
S = S./repmat(sqrt(diag(S'*S))',3,1);

Ap = sparse([[zeros(1,3*nC),-1];[-M,zeros(size(M,1),1)]]);
cp = sparse(size(M,1)+1,1);
qp = size(M,1)+1;

% optimization starts

toUse = 1:size(M,1)/2;
toUseM = sort([2*toUse-1,2*toUse]);
toUseM3 = sort([3*toUse-2,3*toUse-1,3*toUse]);
noUse = [];
mRepr = 3;
radius = 50;
reprs = exp(-1).*ones(size(Ap(2:end,:),1),1);
reprs3 = exp(-1).*ones(length(toUseM3),1);
iter = 1;
wr=wri;
  while((iter <= 1) && (~isempty(toUse)))
     
      reprs = repmat(reprs,1,size(Ap,2));
      S = (((reprs3).*reshape(S,3*size(S,2),1))'*C(toUseM3,:))';
        if(wr > 0)
            bd = sparse([(2/3).*S;-1;-1]);
            K.q = [qc,length(toUseM)+1,ql];
            [x,y,info]=sedumi([[Ac,sparse(size(Ac,1),2)];[[Ap(1,:);reprs.*Ap(toUseM+1,:)],sparse(length(toUseM)+1,1)];[[Al(1,1:end-1);Wr*Al(2:end,1:end-1)],sparse(size(Al,1),1),Al(:,end)]],bd,[cc;[cp(1);cp(toUseM+1)];[cl(1);Wr*cl(2:end)]],K,pars);
        else
      
      
      bd = sparse([(2/3).*S;-1]);
      K.q = [qc,length(toUseM)+1];
      [x,y,info]=sedumi([[Ac,sparse(size(Ac,1),1)];[Ap(1,:);reprs.*Ap(toUseM+1,:)]],bd,[cc;[cp(1);cp(toUseM+1)]],K,pars)
        end
      
      
      Cc = reshape(y(1:3*nC),3,nC);
      [repr,reprs] = compute_reprojection_error(Cc',b,mesh.faces(v,:)-1,p',A);
      mRepr = max(reprs(toUse));
      toUse = find(reprs < radius);
      if(~isempty(toUse))
          toUseM = sort([2*toUse-1;2*toUse]);
          toUseM3 = sort([3*toUse-2;3*toUse-1;3*toUse]);
          noUse = find(reprs >= radius);
          med = median(reprs(toUse));
          if(med > 0)
              reprs = exp(-reprs./med);
          else
              reprs = exp(-1)*ones(length(reprs),1);
          end
          reprs3 = reshape([reprs(toUse),reprs(toUse),reprs(toUse)]',3*length(toUse),1);
          reprs = reshape([reprs(toUse),reprs(toUse)]',2*length(toUse),1);
          
          S = inv(A)*uvs(:,toUse);
          S = S./repmat(sqrt(diag(S'*S))',3,1);
          
          lerr = compute_length_error(Cc',E,L);
          fprintf('Mean Error: %d, Max Error: %d, Nb Corr: %d, Radius: %d, Length Err: %d\n',repr,mRepr,length(toUse),radius,lerr);
          if(wr > 0)
              Wr = compute_patches_weights(Nr,noUse,nM,wr);
          end
      end
      iter = iter+1;
      radius = 0.5*radius;
  
      
      
  end
  meshOut=mesh;
  meshOut.vertexPos=Cc';
  P=b2p(meshOut.vertexPos,mesh.faces(v,:),b);   
end



