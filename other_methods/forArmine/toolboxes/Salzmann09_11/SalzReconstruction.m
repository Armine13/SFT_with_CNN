function [meshOut,P]=SalzReconstruction(p,v,b,mesh,A)
% p 2xN points detected in the image
% v face index for each point p in the template
% b barycentric coordinates of the points p nx3 matrix
%mesh.faces ---->Faces of the mesh
%mesh.B     ----> Barycentrics of the mesh
%mesh.vertexPos ----> Vertexes of the mesh
% A is the intrinsic calibration matrix
% meshOut is the triangulated mesh


tri=mesh.faces-1;
CwR=mesh.vertexPos;
E = compute_edges2(tri,size(CwR,1));
L = zeros(size(E,1),1);
for i=1:length(L)
    t = CwR(E(i,2),:) - CwR(E(i,1),:);
    L(i) = sqrt(t*t');
end
nC = size(CwR,1);
pars=[];
pars.fid=1;

% Precompute socp length constraints
[Ac,cc,qc] = socpConstraints([E,L],nC);
 
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
radius = 20; %20
reprs = exp(-1).*ones(size(Ap(2:end,:),1),1);
reprs3 = exp(-1).*ones(length(toUseM3),1);
iter = 1;
wr=0;
  while((iter <= 1) && (~isempty(toUse)))
     
      reprs = repmat(reprs,1,size(Ap,2));
      S = (((reprs3).*reshape(S,3*size(S,2),1))'*C(toUseM3,:))';
      bd = sparse([(2/3).*S;-1]);
      K.q = [qc,length(toUseM)+1];
      [x,y,info]=sedumi([[Ac,sparse(size(Ac,1),1)];[Ap(1,:);reprs.*Ap(toUseM+1,:)]],bd,[cc;[cp(1);cp(toUseM+1)]],K,pars);
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



