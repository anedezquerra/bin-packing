param n_figures > 0;
param n_points > 0;
param n_dimensions > 0; 

param geometric_conservation symbolic;
param container_type symbolic;

# Ratio bounds for cylinder: radius : height ≈ 1:3
param ratio_low := 2.5;    # Lower bound (height >= 2.5 * radius)
param ratio_high := 3.5;   # Upper bound (height <= 3.5 * radius)

# Sets definition
set figures := 1 .. n_figures by 1;
set points := 1 .. n_points by 1;
set dimensions := 1 .. n_dimensions by 1;

# Parameters setting
param distances{figures, points, points} >= 0;
param volume > 0;
param total_area > 0;
param softness > 0;

# Variables declaration
var coords{figures, points, dimensions};

# Cylinder-specific variables
var radius >= 0;      # Cylinder radius
var z_min;                # Minimum z-coordinate
var z_max;                # Maximum z-coordinate
var height = z_max - z_min;  # Cylinder height

# Existing variables
var alpha{figures, figures};
var beta{figures, figures};
var v{figures, figures, dimensions};
var tetrahedron_volume{figures} >= 0;
var tetrahedron_surface_area{figures} >= 0;

# Objective: Minimize cylinder volume (π*r²*h, ignore constant π)
minimize Cylinder_Volume: radius^2 * height;

# Cylinder containment constraints
subject to cylinder_radial {i in figures, j in points}:
    coords[i, j, 1]^2 + coords[i, j, 2]^2 <= radius^2;

subject to cylinder_z_min {i in figures, j in points}:
    coords[i, j, 3] >= z_min;

subject to cylinder_z_max {i in figures, j in points}:
    coords[i, j, 3] <= z_max;

# Height definition
subject to height_def: 
    height = z_max - z_min;

# Maintain radius-to-height ratio (approx 1:3)
subject to height_ratio_lower:
    height >= ratio_low * radius;

subject to height_ratio_upper:
    height <= ratio_high * radius;

# --- Existing constraints (corrected syntax) ---
subject to alpha_beta_constraints{i in figures, j in figures: j > i}:
    alpha[i, j] + beta[i,j] <= 0;

subject to alpha_hyperplanes {l in points, i in figures, j in figures: j > i}:
   sum{n in dimensions} v[i,j,n] * coords[i, l, n] <= alpha[i, j];

subject to beta_hyperplanes {l in points, i in figures, j in figures: j > i}:
   sum{n in dimensions} v[i,j,n] * coords[j, l, n] >= -beta[i, j];

subject to equality_constraints{i in figures, j in figures: j > i}:
    sum{n in dimensions} v[i, j, n] ^ 2 = 1;

# Corrected constraint indexing (fixed parentheses)
subject to euclidean_distance_constraints_1_plus {i in figures, l in points: l < card(points)-1}:
    sum{n in dimensions} (coords[i, l, n] - coords[i, l+1, n]) ^ 2 <= distances[i, l, l+1]^2 * (1 + softness);

subject to euclidean_distance_constraints_1_minus {i in figures, l in points: l < card(points)-1}:
    sum{n in dimensions} (coords[i, l, n] - coords[i, l+1, n]) ^ 2 >= distances[i, l, l+1]^2 * (1 - softness);

subject to euclidean_distance_constraints_2_plus {i in figures}:
    sum{n in dimensions} (coords[i, card(points)-1, n] - coords[i, 1, n]) ^ 2 <= distances[i, card(points)-1, 1]^2 * (1 + softness);
    
subject to euclidean_distance_constraints_2_minus {i in figures}:
    sum{n in dimensions} (coords[i, card(points)-1, n] - coords[i, 1, n]) ^ 2 >= distances[i, card(points)-1, 1]^2 * (1 - softness);

subject to euclidean_distance_constraints_3_plus {i in figures, l in points: l < card(points)}:
    sum{n in dimensions} (coords[i, l, n] - coords[i, card(points), n]) ^ 2 <= distances[i, l, card(points)]^2 * (1 + softness);
    
subject to euclidean_distance_constraints_3_minus {i in figures, l in points: l < card(points)}:
    sum{n in dimensions} (coords[i, l, n] - coords[i, card(points), n]) ^ 2 >= distances[i, l, card(points)]^2 * (1 - softness);

subject to volume_constraints{i in figures}:
    tetrahedron_volume[i]
    = abs(
        (1/6)
        * (
            (coords[i,2,1] - coords[i,1,1])
            * (
                (coords[i,3,2] - coords[i,1,2])
                * (coords[i,4,3] - coords[i,1,3])
              - (coords[i,4,2] - coords[i,1,2])
                * (coords[i,3,3] - coords[i,1,3])
              )
          - (coords[i,2,2] - coords[i,1,2])
            * (
                (coords[i,3,1] - coords[i,1,1])
                * (coords[i,4,3] - coords[i,1,3])
              - (coords[i,4,1] - coords[i,1,1])
                * (coords[i,3,3] - coords[i,1,3])
              )
          + (coords[i,2,3] - coords[i,1,3])
            * (
                (coords[i,3,1] - coords[i,1,1])
                * (coords[i,4,2] - coords[i,1,2])
              - (coords[i,4,1] - coords[i,1,1])
                * (coords[i,3,2] - coords[i,1,2])
              )
          )
      );

subject to volume_equality_constraint{i in figures}:
    tetrahedron_volume[i] = volume;