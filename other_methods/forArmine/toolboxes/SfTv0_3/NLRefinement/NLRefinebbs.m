function [ out ] = NLRefinebbs( bbs, ctrlpts, options, p, q, proi, bbsDn, ctrlptsDn )
% refine and return the final \varphi function

global iterno
iterno = 0;
lambdas = options.phi.er*ones(options.phi.nC-3, options.phi.nC-3);
bendmat = bbs_bending(bbs, lambdas);
[Ub,Sb,Vb] = svd(full(bendmat));
sqrtbending = sqrt(Sb)*Vb';
sqrtbending = sparse(sqrtbending);

% Jdeltau = bbs_eval(bbsDn,ctrlptsDn,proi(1,:),proi(2,:),1,0);
% Jdeltav = bbs_eval(bbsDn,ctrlptsDn,proi(1,:),proi(2,:),0,1);
% Jdelta = [Jdeltau; Jdeltav];

Jdeltau = [ones(1,size(proi,2));zeros(2,size(proi,2))];
Jdeltav = [zeros(1,size(proi,2));ones(1,size(proi,2));zeros(1,size(proi,2))];
Jdelta = [Jdeltau; Jdeltav];

npts = length(proi);
ctrlpts=ctrlpts';
opt = optimoptions('lsqnonlin','Jacobian','on','DerivativeCheck','off','MaxIter',options.maxiter);
ctrlptsv = lsqnonlin(@(param)NLcost(bbs,param,options,p,q,proi,Jdelta,sqrtbending), ...
           ctrlpts(:),[],[],opt);

ctrlptsr = reshape(ctrlptsv,[],3);       
out.phi.bbs = bbs;
out.phi.ctrlpts = ctrlptsr';
out.phi.options = options;
end

