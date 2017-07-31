function [L,nreig,ncombinations]=getcombinations(I,th)
        P=find(I>th);
        I(P)=1;
        P=find(I<=th);
        I(P)=0;        
        [L,nreig]=bwlabel(I,8);
        
        

        listr=[];
        for i=[1:nreig]
            
            length(find(L==i));
            if(length(find(L==i))<9*9)
                I(find(L==i))=0;
                L(find(L==i))=0;            
            else
            listr=[i,listr];               
            end

        end
        L2=L;
        L(:)=0;
        for i=[1:length(listr)]
            L(L2==listr(i))=i;
        end
        nreig=length(listr);
         [Ln,nreign]=bwlabel(~I,8);

         listrn=[];
        for i=[1:nreign]
            
            length(find(Ln==i));
            if(length(find(Ln==i))<9*9)
                I(find(Ln==i))=1;
                Ln(find(Ln==i))=0;            
            else
            listrn=[i,listrn];               
            end

        end
        L2n=Ln;
        Ln(:)=0;
        for i=[1:length(listrn)]
            Ln(L2n==listrn(i))=i;
        end
        nreign=length(listrn);
       
         Ln(find(Ln>0))=Ln(find(Ln>0))+nreig;
         L=L+Ln;
        nreig=nreig+nreign;
        ncombinations=dec2bin(0:2^nreig-1);

      