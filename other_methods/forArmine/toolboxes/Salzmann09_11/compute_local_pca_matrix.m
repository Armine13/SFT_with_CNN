function [P,Vr] = compute_local_pca_matrix(patches,wvs,nM,PC,Vars,Cc0)

nC = length(wvs);
nL = size(patches,2);
nP = size(patches,1);

patches = 1+patches*3;
patches = [patches, patches+1, patches+2];
patches = sort(patches,2);

Vr = zeros(nP*nM,3*nC+1);

if(nM < nL)
    P = zeros(nP*3*nL,3*nC+1);
    PPt = PC(:,1:nM)*PC(:,1:nM)';
    ImPPt = eye(3*nL) - PPt;
    S = sqrt(diag(1./Vars(1:nM)));
    SPt = S*PC(:,1:nM)';
    sigm = 1;

    wvs = repmat(wvs,3,1);
    wvs = reshape(wvs,1,3*nC);
    wvs = sqrt(wvs);

    for i=1:nP
        P((i-1)*3*nL+1:i*3*nL,patches(i,:)) = ImPPt;
        P((i-1)*3*nL+1:i*3*nL,end) = -ImPPt*Cc0(patches(i,:));
        P((i-1)*3*nL+1:i*3*nL,:) = (1/sigm).*(diag(wvs(patches(i,:)))*P((i-1)*3*nL+1:i*3*nL,:));

        Vr((i-1)*nM+1:i*nM,patches(i,:)) = SPt;
        Vr((i-1)*nM+1:i*nM,end) = -SPt*Cc0(patches(i,:));
    end
else
    P = [];
    S = sqrt(diag(1./Vars(1:nM)));
    SPt = S*PC(:,1:nM)';
    
    for i=1:nP
        Vr((i-1)*nM+1:i*nM,patches(i,:)) = SPt;
        Vr((i-1)*nM+1:i*nM,end) = -SPt*Cc0(patches(i,:));
    end
end
