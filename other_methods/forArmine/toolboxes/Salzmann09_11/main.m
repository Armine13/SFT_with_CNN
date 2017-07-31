A = load('data/cam.intr');
Rt = load('data/cam.ext');
CwR = load('data/mesh.pts');
tri = load('data/mesh.tri');
[patches,wvs] = make_patches(0,0,5,5,9,9,1);
E = compute_edges(tri,size(CwR,1));
L = zeros(size(E,1),1);
for i=1:length(L)
    t = CwR(E(i,2),:) - CwR(E(i,1),:);
    L(i) = sqrt(t*t');
end
nC = size(CwR,1);

wri = 10;
if(wri > 0)    
    Var = load('data/modes.var');
    PC = load('data/modes.mds');
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
pars.fid=0;
[Ac,cc,qc] = socpConstraints([E,L],nC);

for f=1       
    tmp = sprintf('data/matches/matches_noise_%d.txt',f);
    
    matches_all = load(tmp);
    
    wr = wri;
    if(wr > 0)
        Nr = compute_patches_matches(matches_all,patches);
        Wr = wr*speye(nM*size(Nr,2));
    end
    
    M = compute_correspondence_matrix(A,matches_all(:,4:6),matches_all(:,1:3),matches_all(:,7:8),1,nC);
    
    uvs = [matches_all(:,7:8)';ones(1,size(matches_all,1))];
    S = inv(A)*uvs;
    C = sparse(getVToXMatrix(matches_all(:,4:6),matches_all(:,1:3),nC));
    S = S./repmat(sqrt(diag(S'*S))',3,1);
    
    Ap = sparse([[zeros(1,3*nC),-1];[-M,zeros(size(M,1),1)]]);
    cp = sparse(size(M,1)+1,1);
    qp = size(M,1)+1;
    
    toUse = 1:size(M,1)/2;
    toUseM = sort([2*toUse-1,2*toUse]);
    toUseM3 = sort([3*toUse-2,3*toUse-1,3*toUse]);
    noUse = [];
    mRepr = 3;
    radius = 50;
    reprs = exp(-1).*ones(size(Ap(2:end,:),1),1);
    reprs3 = exp(-1).*ones(length(toUseM3),1);
    iter = 1;
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
            [x,y,info]=sedumi([[Ac,sparse(size(Ac,1),1)];[Ap(1,:);reprs.*Ap(toUseM+1,:)]],bd,[cc;[cp(1);cp(toUseM+1)]],K,pars);
        end
        
        Cc = reshape(y(1:3*nC),3,nC);
        [repr,reprs] = compute_reprojection_error(Cc',matches_all(:,4:6),matches_all(:,1:3),matches_all(:,7:8),A);
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
    
    Cw = inv([Rt;[0,0,0,1]])*[Cc;ones(1,size(Cc,2))];
    Cw = Cw(1:3,:)';
    if(wr > 0)
        tmp = sprintf('data/res/model/frame_%d.pts',f);
    else
        tmp = sprintf('data/res/no_model/frame_%d.pts',f);
    end
    save(tmp,'Cw','-ASCII');
end
