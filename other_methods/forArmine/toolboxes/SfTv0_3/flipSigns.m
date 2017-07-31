% Function to flip signs of the analytical jacobian according to TPS
% jacobian.


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

