function [indices,tellapsed]=DetectOutliers(x1,y1,x2o,y2o,Neig,mu,lambda,ls,th)
%Debug parameters !!. Not documented.
flag.step1=0;flag.step2=0;flag.step3=1;flag.step4=4;flag.local=1;
if(nargin<9)
    th=20;
    %th=0.15;
end
% Generate spline control points in a reference frame [0,1;1,0];
BoundPoints=[0,0;1,0;1,1;0,1];
[C,linelist]=generate_centers(ls,BoundPoints);
Epsilon_Lambda=EpsilonLambda(C,lambda);
tic;
if(flag.local==0)
    smoothlocal=LocalSmoothness(x1,y1,x2o,y2o,Neig,mu,lambda,ls,C,Epsilon_Lambda,flag.step1);
else
    smoothlocal=LocalSmoothnessLocal(x1,y1,x2o,y2o,Neig,mu,lambda,ls,C,Epsilon_Lambda);
end

m=length(x1);


if(flag.step3==1)
    for iter=1:flag.step4
        
        x1t=x1(find(smoothlocal<=th));
        y1t=y1(find(smoothlocal<=th));
        x2ot=x2o(find(smoothlocal<=th));
        y2ot=y2o(find(smoothlocal<=th));
        mt=length(x1t);
        if(mt>2)
            for i=[1:m]
                %if(isnan(smoothlocal(i)))
                %    continue;
                %end
                if(smoothlocal(i)>th)
                    
                    TRIt=delaunay([x1t;x1(i)],[y1t;y1(i)]);
                    %TRIt=filter_triangulation(TRIt,[x1t;x1(i)],[y1t;y1(i)]);
                    Neigt=neighbour_list(TRIt,mt+1);
                    if(iter==flag.step4)
                        flagp=flag.step2;
                    else
                        flagp=0;
                    end
                    if(flag.local==0)
                        [norma,qtemp]=LocalSmoothness([x1t;x1(i)],[y1t;y1(i)],[x2ot;x2o(i)],[y2ot;y2o(i)],Neigt,mu,lambda,ls,C,Epsilon_Lambda,flagp,mt+1);
                    else
                        [norma,qtemp]=LocalSmoothnessLocal([x1t;x1(i)],[y1t;y1(i)],[x2ot;x2o(i)],[y2ot;y2o(i)],Neigt,mu,lambda,ls,C,Epsilon_Lambda,mt+1);
                    end
                    
                    if(norma(mt+1)<smoothlocal(i))
                        smoothlocal(i)=norma(mt+1);
                    end
                    
                    
                end
                
            end
        end
    end
end
indices=find(smoothlocal<=th);
tellapsed=toc;
