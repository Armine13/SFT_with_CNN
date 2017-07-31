function [ E, JE ] = NLcost( bbs, paramts, options, p, q, proi, Jdelta, sqrtbending)
% Cost function with Jacobian for NL refinement
global iterno
iterno = iterno+1;
lbdiso = options.lbdiso;
% lbdbend = options.lbdbend;
lbdrep = 9e1;
npts = length(proi);
np = length(p);

ctrlpts = reshape(paramts,[],3);
ctrlpts = ctrlpts';
Qest = bbs_eval(bbs, ctrlpts, p(1,:),p(2,:),0,0);
qest = Qest./repmat(Qest(3,:),3,1);

% Reprojection error
Ed = q- qest(1:2,:);
Ed =lbdrep* Ed(:);

% Smoothness error
Es = sqrtbending*ctrlpts';
Es = Es(:);

% Isometric error
Jphiu = bbs_eval(bbs, ctrlpts, proi(1,:),proi(2,:),1,0);
Jphiv = bbs_eval(bbs, ctrlpts, proi(1,:),proi(2,:),0,1);

Jdeltau = Jdelta(1:3,:);
Jdeltav = Jdelta(4:6,:);

Ei1 = Jphiu(1,:).^2+Jphiu(2,:).^2+Jphiu(3,:).^2- Jdeltau(1,:).^2 - Jdeltau(2,:).^2 ...
     - Jdeltau(3,:).^2;
Ei2 = 2*(Jphiu(1,:).*Jphiv(1,:) + Jphiu(2,:).*Jphiv(2,:) + Jphiu(3,:).*Jphiv(3,:)- ...
    Jdeltau(1,:).*Jdeltav(1,:) -Jdeltau(2,:).*Jdeltav(2,:)) -Jdeltau(3,:).*Jdeltav(3,:);
Ei3 = Jphiv(1,:).^2 + Jphiv(2,:).^2 + Jphiv(3,:).^2 - Jdeltav(1,:).^2 - Jdeltav(2,:).^2 ...
    -Jdeltav(3,:).^2;

Ei = lbdiso*[Ei1';Ei2';Ei3'];

% Jphi = [Jphiu; Jphiv];
coloc = bbs_coloc(bbs, p(1,:), p(2,:));

% Ed derivatives
Jd = zeros(2*np,3*length(ctrlpts));
% Jd = [];
for i = 1: np
    qi = qest(1:2,i);
    pi = p(1:2,i);
    phizi = Qest(3,i);
    dphiL = blkdiag(coloc(i,:),coloc(i,:),coloc(i,:));
    Jd(i*2-1:2*i,:) = lbdrep*[-1/phizi*[eye(2) -qi]*dphiL];        
end

% Es derivatives
Js = blkdiag(sqrtbending,sqrtbending,sqrtbending);

% Ei derivatives
colocroi = bbs_coloc(bbs, proi(1,:), proi(2,:));
colocdu = bbs_coloc_deriv(bbs, proi(1,:), proi(2,:),1,0);
colocdv = bbs_coloc_deriv(bbs, proi(1,:), proi(2,:),0,1);

% Ei1 = Jphiu(1,:).^2 + Jphiu(2,:).^2 + Jphiu(3,:).^2-Jdeltau(1,:).^2 + Jdeltau(2,:).^2 ;
% Ei2 = 2*(Jphiu(1,:).*Jphiv(1,:) + Jphiu(2,:).*Jphiv(2,:) + Jphiu(3,:).*Jphiv(3,:)- ...
%     Jdeltau(1,:).*Jdeltav(1,:) +Jdeltau(2,:).*Jdeltav(2,:));
% Ei3 = Jphiv(1,:).^2 + Jphiv(2,:).^2 + Jphiv(3,:).^2 - Jdeltav(1,:).^2 + Jdeltav(2,:).^2 ;

% Jphiu(1,:).^2+Jphiu(2,:).^2-Jdeltau(1,:).^2 + Jdeltau(2,:).^2 ;
ncol = size(colocdu,2);
JEi1 = zeros(length(proi),size(ctrlpts,2)*3);
JEi2 = zeros(length(proi),size(ctrlpts,2)*3);
JEi3 = zeros(length(proi),size(ctrlpts,2)*3);
% JEi1 = [];
% JEi2 = [];
% JEi3 = [];

for i = 1: length(proi)
    Jphiui = Jphiu(:,i);
    Jphivi = Jphiv(:,i);
    
    Jdeltaui = Jdeltau(:,i);
    Jdeltavi = Jdeltav(:,i);
    
    JEi1(i,:) =  2*Jphiui(1)*[colocdu(i,:), zeros(1,2*ncol)] + ...
    2*Jphiui(2)*[zeros(1,ncol), colocdu(i,:), zeros(1,ncol)] + ...
    2*Jphiui(3)*[zeros(1,2*ncol), colocdu(i,:)] ;


    JEi3(i,:) = 2*Jphivi(1)*[colocdv(i,:), zeros(1,2*ncol)] + ...
    2*Jphivi(2)*[zeros(1,ncol), colocdv(i,:), zeros(1,ncol)] + ...
    2*Jphivi(3)*[zeros(1,2*ncol), colocdv(i,:)] ;
    
    JEi2(i,:) = 2*Jphiui(1)*[colocdv(i,:), zeros(1,2*ncol)] +...
        2*Jphivi(1)*[colocdu(i,:), zeros(1,2*ncol)] +...
    2*Jphiui(2)*[zeros(1,ncol), colocdv(i,:), zeros(1,ncol)] + ...
    2*Jphivi(2)*[zeros(1,ncol), colocdu(i,:), zeros(1,ncol)] + ...
    2*Jphiui(3)*[zeros(1,2*ncol), colocdv(i,:)] + ...
    2*Jphivi(3)*[zeros(1,2*ncol), colocdu(i,:)] ;
end

JEi = lbdiso*[JEi1;JEi2;JEi3];

E = [Ed;Es;Ei];
JE = [Jd;Js;JEi];
% JE = [1e4*Jd;Js];
% E = [1e4*Ed;Es];
if options.verbose ~=0
    disp(sprintf('Iter %d: Reproj cost=%f, Inext cost=%f, Bending cost= %f\n',iterno,norm(Ed).^2,norm(Ei).^2,norm(Es).^2));
end    

end

