%% Shape from Template using physics & position based model
function shape = SfTerol( xo, template, pixels, options )
%% input
% template [3xN]: 3D point coordinates of the template (in meters)  
% pixels [2xN]: corresponding image coordinates (in pixels)
% options : parametrs for the algorithm, e.g. intrinsic camera matrix K...
%% output
% shape [3xN]: estimated 3D shape point coodinates (in meters)
%% shape from template algorithm
K                   = options.K;  % intrinsic camera matrix
mass                = options.mass; % template weight (kg)
k_damping           = options.k_damping; % template velocity damping coefficient between [0...1]  (default 0.01)
k_stretch           = options.k_stretch; % template stretching stiffness coefficient between [0...1] (isometric case = 1)
k_bend              = options.k_bend; % template stretching stiffness coefficient between [0...1] 
maxIter             = options.maxIter; % maximum number of loop iterations (default 5000)
n_solverIterations  = options.solverIter; % number of solver iterations
eps                 = options.threshold; % minimum displacement threshold to stop the loop  (default 10e-7)


%% INiTiALISATION (physics & position based model)
%% initialize particles
n_particles = size(template,2);    % number of particles
particle_mass = mass/n_particles;  % particle mass

particles.pos  = template;   % current configuration
particles.pose = template;   % estimated configuration
particles.pos_prev = template;   % previous configuration 

particles.vel = zeros(size(particles.pos));    %  particle velocities
particles.force = zeros(size(particles.pos));  %  particle forces
particles.mass = repmat( particle_mass, [1, n_particles] ); % particle masses

fx = K(1,1);  fy = K(2,2);
uo = K(1,3);  vo = K(2,3);
m = pixels - repmat( [uo;vo], [1, n_particles] );
xy = m ./ repmat( [fx;fy], [1, n_particles] ); 
xy = [xy; ones(1, n_particles) ]; 
xy_norm = sqrt(sum(xy.^2,1));

particles.unit = xy ./ repmat( xy_norm, [3,1] );  % unit sight-line vectors of particles

%% initialize stretching constraints (connection points and initial lengths from irregular data triangulation)
Vertices = particles.pos';
tri = delaunay(Vertices(:,1:2));
edge_list = unique(sort([tri(:,2), tri(:,1); ...
                         tri(:,3), tri(:,2); ...
                         tri(:,1), tri(:,3)],2),'rows');
         
stretching_constraints.connection_list = edge_list;

p1 = Vertices( edge_list(:,1), : )';
p2 = Vertices( edge_list(:,2), : )';

stretching_constraints.length0 = sqrt(sum((p2-p1).^2,1)); 


%% initialize bending constraints (connection points and initial lengths)      
tri = sort(tri,2);
nt = size(tri,1);
trinum = repmat((1:nt)',3,1);

edges = [tri(:,[1 2]);tri(:,[1 3]);tri(:,[2 3])];
[edges,tags] = sortrows(edges);

% and shuffle the numbers of each triangle
trinum = trinum(tags);

% adjacent triangles list
k = find( all( diff(edges,1) == 0, 2 ) );
tri_list = trinum([k,k+1]);
tri_list = sortrows(tri_list);

% create bending constraints 
for i = 1:size(tri_list,1)

    t1 = tri(tri_list(i,1),:); 
    t2 = tri(tri_list(i,2),:);

    iP3 = setdiff(t1,t2);
    iP4 = setdiff(t2,t1);
        
    bending_constraints(i).connection_list = [ iP3, iP4 ];
        
    p3 = Vertices(iP3,:)';
    p4 = Vertices(iP4,:)';
    
    bending_constraints(i).length0 = norm(p4-p3);        
    
end


%% LOOP  LOOP  LOOP  LOOP  LOOP  LOOP  LOOP
ts = 1;  % sampling time delta t

n_stretching_constraints = length(stretching_constraints.length0);
n_bending_constraints = size(bending_constraints,2);

%% initialize model
particles.pos  = xo;   % current configuration
particles.pose = xo;   % estimated configuration
particles.pos_prev = xo;   % previous configuration 


%% START RECONSTRUCTION
for iter = 0:maxIter 
 
%% apply external forces here and then update velocities 
% ???

%% damp velocities
  particles.vel = particles.vel - k_damping*particles.vel;

%% estimate new positions
particles.pose = particles.pos  + ts*particles.vel;

%% apply sight-line constraints 
  particles.pose = repmat( sum(particles.pose.*particles.unit ), [3,1] ) .* particles.unit;

%% solver  
  for solverIter = 1:n_solverIterations
      
%% apply stretching constraints  
  
      for i=1:n_stretching_constraints
   
        i1 = stretching_constraints.connection_list(i,1);
        i2 = stretching_constraints.connection_list(i,2);
        d = particles.pose(:,i2) - particles.pose(:,i1);
        d_norm = norm(d);
        d_unit = d / d_norm;
        lo = stretching_constraints.length0(i);    
        
        e1 = k_stretch*0.5*( d_norm - lo )*d_unit;
        e2 = -e1;
              
        particles.pose(:,i1) =  particles.pose(:,i1) + e1; 
        particles.pose(:,i2) =  particles.pose(:,i2) + e2; 

     end
     
%% apply bending constraints  
 
    for i=1:n_bending_constraints
    
        i1 = bending_constraints(i).connection_list(1);
        i2 = bending_constraints(i).connection_list(2);
                
        d = particles.pose(:,i2) - particles.pose(:,i1);
        d_norm = norm(d);
        d_unit = d/d_norm;
        lo = bending_constraints(i).length0;
        
        e1 = k_bend*0.5*( d_norm - lo )*d_unit;
        e2 = -e1;
             
        particles.pose(:,i1) =  particles.pose(:,i1) + e1;
        particles.pose(:,i2) =  particles.pose(:,i2) + e2;
        
    end
    
  end % end of solver loop
 
%% update particle states 
   particles.pos_prev = particles.pos;
   particles.vel = (particles.pose - particles.pos)/ts;
   particles.pos = particles.pose; 

%% break loop condition (while the mesh moves no more than the threshold RMSE value)
   if( sqrt( mean(sum( (particles.pos_prev - particles.pos).^2)) ) < eps) disp('breaking loop'); iter, break; end

end

iter,
%% LOOP ENDS
shape = particles.pos;

if( shape(3,1) < 0 ) shape = -shape; end

