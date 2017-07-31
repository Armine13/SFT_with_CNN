function [b1 b2 b3 ind_tri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z, x, y, z, ind_tri)
%RIS_CART_TO_BARY Converts cartesian coordinates to barycentric ones.
%
% SYNTAX
%  [b1 b2 b3] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z, x, y, z)
%    Computes both the barycentric coordinates and the facets to which
%    belongs the points.
%
%  [b1 b2 b3 ind_tri] = ris_cart_to_bary(tri, tri_x, tri_y, tri_z, x, y, z, ind_tri)
%    Computes only the barycentric coordinates, considering that we already
%    known to which facet belongs a points (typically because "ind_tri" has
%    already been computed before, using the other variant of this 
%    function).
%
% DESCRIPTION
%  Converts cartesian coordinates to barycentric ones.
%
% INPUT ARGUMENTS
%
%  - tri, tri_x, tri_y, tri_z: triangular mesh
%     * tri [nx3 matrix]: tri(i,:) are the 3 indices of the vertices of the
%         i-th facet.
%     * tri_x, tri_y, tri_z [px1 matrices]: cartesian coordinates of the 
%         vertices.
%  
%  - x, y, z [matrices]: cartesian coordinates of the points to convert.
%
%  - ind_tri [matrix, same size as x, y and z]: ind_tri(i) is the indices
%      of the row in "tri" (ie the n° of the facet) to which the point
%      (x(i),y(i),z(i)) belongs to.
%      This parameter cannot be passed to the function if "ind_tri" is in
%      the output arguments.
%
% OUTPUT ARGUMENTS
%
%  - b1, b2, b3 [matrices, same size as x, y and z]: (b1(i),b2(i),b3(i)) 
%      are the barycentric coordinates of the point (x(i),y(i),z(i)).
%
%  - ind_tri [optional, matrix, same size as x, y and z]: ind_tri(i) is the
%      indices of the row in "tri" (ie the n° of the facet) to which the
%      point (x(i),y(i),z(i)) belongs to.
%      This parameter cannot be returned by the function if "ind_tri" is in
%      the input arguments.
%
% (c)2010, Florent Brunet

% RIS is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% RIS is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
% for more details.
% 
% You should have received a copy of the GNU General Public License along
% with this program; if not, write to the Free Software Foundation, Inc.,
% 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

if (nargin == 8) && (nargout == 4)
    error('ris_cart_to_bary:param', '"ind_tri" cannot be appear as an input and an output argument at the smae time');
elseif (nargin == 7) && (nargout == 4)
    %% Compute both the barycentric coordinates and the facets to which
    %  belongs the points

    % Number of facets
    n_tri = size(tri, 1);

    ind_tri = zeros(size(x));
    b1 = zeros(size(x));
    b2 = zeros(size(x));
    b3 = zeros(size(x));

    for i = 1:numel(x)
        if mod(i, 1000) == 0
            disp(i)
        end
        found = false;
        for j = 1:n_tri
            M = [tri_x(tri(j,:))' ; tri_y(tri(j,:))' ; tri_z(tri(j,:))'];
            bar = M \ [x(i) ; y(i) ; z(i)];
            if (all(bar>=-0.01))
                ind_tri(i) = j;
                b1(i) = bar(1);
                b2(i) = bar(2);
                b3(i) = bar(3);
                found = true;
                break;
            end
        end
        if ~found
            error('ris_cart_to_bary:bad_point', 'It was impossible to compute the barycentric coordinates of a point');
        end
    end
elseif (nargin == 8) && (nargout == 3)
    %% We just compute the barycentric coordinates
    b1 = zeros(size(x));
    b2 = zeros(size(x));
    b3 = zeros(size(x));

    for i = 1:numel(x)
        ind = tri(ind_tri(i),:);
        M = [tri_x(ind)' ; tri_y(ind)' ; tri_z(ind)'];
        bar = M \ [x(i) ; y(i) ; z(i)];
        b1(i) = bar(1);
        b2(i) = bar(2);
        b3(i) = bar(3);
    end
else
    error('ris_cart_to_bary:number_param', 'Wrong number of argument');
end