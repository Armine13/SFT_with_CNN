function [tri tri_x tri_y tri_z] = ...
    ris_create_tri_mesh(xmin, xmax, nptsx, ...
                        ymin, ymax, nptsy, ...
                        val_z)
%RIS_CREATE_TRI_MESH Create a triangular mesh.
%
% SYNTAX
%   [tri tri_x tri_y tri_z] = ...
%       ris_create_tri_mesh(xmin, xmax, nptsx, ...
%                           ymin, ymax, nptsy)
%    Assume val_z = 1
%
%   [tri tri_x tri_y tri_z] = ...
%       ris_create_tri_mesh(xmin, xmax, nptsx, ...
%                           ymin, ymax, nptsy, ...
%                           val_z)
%
% DESCRIPTION
%  Creates triangular meshes similar to the ones displayed in the paper 
%  "M. Salzmann, P. Fua. Reconstructing Sharply Folding Surfaces: A Convex 
%   Formulation. CVPR, 2009."
%
% INPUT ARGUMENTS
%  - Definition domain of the mesh [xmin,xmax] x [ymin,ymax]
%  - Mesh size: nptsx x nptsy
%
% OUTPUT ARGUMENTS
%  - tri [nf x 3 matrix]: tri(i,:) are the 3 indices of the vertices of the
%      i-th facet.
%  - tri_x, tri_y, tri_z [nv x 1 vectors]: cartesian coordinates of the 
%      vertices.
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

if nargin <= 6
    val_z = 1;
end

npts = nptsx * nptsy;

indices = reshape(1:npts, nptsy, nptsx)';

tri = zeros(2*(nptsx-1)*(nptsy-1), 3);
k = 1;
for i_x = 1:nptsx-1
    for i_y = 1:nptsy-1
        tri(k:k+1,:) = ...
            [indices(i_x, i_y) indices(i_x+1, i_y) indices(i_x+1, i_y+1) ; 
             indices(i_x, i_y) indices(i_x+1, i_y+1) indices(i_x, i_y+1) ];
        k = k + 2;
    end
end

[tri_x tri_y] = meshgrid(linspace(xmin, xmax, nptsx), linspace(ymin, ymax, nptsy));
tri_x = tri_x(:);
tri_y = tri_y(:);
tri_z = val_z .* ones(size(tri_x));
