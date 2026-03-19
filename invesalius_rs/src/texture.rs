// Surface texture generation for medical imaging
// Ported from Cython to Rust for InVesalius3 integration

use ndarray::prelude::*;
use ndarray::ArrayView3;
use num_traits::NumCast;

// Re-use interpolation from the existing module
use crate::interpolation::trilinear_interpolate_internal;

/// Calculate barycentric coordinates for UV mapping
#[inline]
fn uv_to_barycentric(
    u0: f64, v0: f64,
    u1: f64, v1: f64,
    u2: f64, v2: f64,
    u: f64, v: f64,
) -> [f64; 3] {
    let denom = (v2 - v0) * (u1 - u0) + (u2 - u0) * (v1 - v0);
    if denom.abs() < 1e-10 {
        return [1.0, 0.0, 0.0];
    }
    
    let bar0 = ((v2 - v0) * (u - u0) + (u2 - u0) * (v - v0)) / denom;
    let bar1 = ((v0 - v1) * (u - u0) + (u0 - u1) * (v - v0)) / denom;
    let bar2 = 1.0 - bar0 - bar1;
    
    [bar0, bar1, bar2]
}

/// Generate texture coordinates and texture image from mesh and volume data
/// 
/// This is the main function that creates a texture atlas from a 3D mesh.
/// It lays out all triangles in a 2D grid and samples volume data at each position.
///
/// # Arguments
/// * `vertices` - Mesh vertices (N x 3)
/// * `normals` - Vertex normals (N x 3)
/// * `faces` - Face indices (M x 3)
/// * `volume` - 3D volume data (D x H x W)
/// * `spacing` - Volume spacing (3)
/// * `window_width` - Window width for visualization
/// * `window_level` - Window level for visualization
/// * `clut` - Color lookup table (256 x 3)
/// * `texture_dim` - Texture atlas dimension (default 5000)
///
/// # Returns
/// * `(tcoords, texture_image, texture_normals)` - UV coordinates, texture image, and normals
pub fn generate_surface_texture_internal<V, F, T>(
    vertices: ArrayView2<V>,
    normals: ArrayView2<V>,
    faces: ArrayView2<F>,
    volume: ArrayView3<T>,
    spacing: &[f64; 3],
    window_width: i32,
    window_level: i32,
    clut: ArrayView2<u8>,
    texture_dim: usize,
) -> (Array2<f64>, Array3<u8>, Array3<u8>)
where
    V: Copy + Into<f64> + Send + Sync,
    F: Copy + TryInto<usize> + Send + Sync,
    T: Copy + Into<f64> + Send + Sync + NumCast,
{
    let n_faces = faces.shape()[0];
    
    // Calculate grid dimensions for texture atlas
    let nx = (n_faces as f64).sqrt() as usize;
    let ny = (n_faces as f64 / nx as f64).ceil() as usize;
    
    let d = texture_dim;
    let dtx = d as f64 / nx as f64;
    let dty = d as f64 / ny as f64;
    
    // Output arrays
    let mut tcoords = Array2::<f64>::zeros((n_faces, 6));
    let mut texture_image = Array3::<u8>::zeros((d, d, 3));
    let mut texture_normals = Array3::<u8>::zeros((d, d, 3));
    
    let offset = 2;
    
    // Process each triangle
    for tc in 0..n_faces {
        let i = tc % nx;
        let j = tc / nx;
        
        // Calculate UV coordinates for the triangle corners
        let c0x = (i as f64 * dtx + offset as f64) as usize;
        let c0y = (j as f64 * dty + offset as f64) as usize;
        let c1x = (c0x as f64 + dtx - offset as f64) as usize;
        let c1y = c0y;
        let c2x = ((c0x as f64 + c1x as f64) / 2.0) as usize;
        let c2y = (c0y as f64 + dty - offset as f64) as usize;
        
        // Store UV coordinates (6 values per triangle: u0,v0,u1,v1,u2,v2)
        tcoords[[tc, 0]] = c0x as f64 / d as f64;
        tcoords[[tc, 1]] = 1.0 - c0y as f64 / d as f64;
        tcoords[[tc, 2]] = c1x as f64 / d as f64;
        tcoords[[tc, 3]] = 1.0 - c1y as f64 / d as f64;
        tcoords[[tc, 4]] = c2x as f64 / d as f64;
        tcoords[[tc, 5]] = 1.0 - c2y as f64 / d as f64;
        
        // Get vertex and normal indices for this face
        let v0_idx = faces[[tc, 0]].try_into().unwrap_or(0);
        let v1_idx = faces[[tc, 1]].try_into().unwrap_or(0);
        let v2_idx = faces[[tc, 2]].try_into().unwrap_or(0);
        
        // Get vertex positions
        let v0 = vertices.row(v0_idx);
        let v1 = vertices.row(v1_idx);
        let v2 = vertices.row(v2_idx);
        
        // Get vertex normals
        let n0 = normals.row(v0_idx);
        let n1 = normals.row(v1_idx);
        let n2 = normals.row(v2_idx);
        
        // Sample volume data for each pixel in the triangle
        for y in (c0y.saturating_sub(1))..(c2y + 1).min(d) {
            for x in (c0x.saturating_sub(1))..(c1x + 1).min(d) {
                // Calculate barycentric coordinates
                let bar = uv_to_barycentric(
                    tcoords[[tc, 0]], tcoords[[tc, 1]],
                    tcoords[[tc, 2]], tcoords[[tc, 3]],
                    tcoords[[tc, 4]], tcoords[[tc, 5]],
                    x as f64 / d as f64,
                    1.0 - y as f64 / d as f64,
                );
                
                // Interpolate vertex positions using barycentric coordinates
                let vx = bar[0] * v0[0].into() + bar[1] * v1[0].into() + bar[2] * v2[0].into();
                let vy = bar[0] * v0[1].into() + bar[1] * v1[1].into() + bar[2] * v2[1].into();
                let vz = bar[0] * v0[2].into() + bar[1] * v1[2].into() + bar[2] * v2[2].into();
                
                // Convert to volume coordinates
                let px = vx / spacing[0];
                let py = vy / spacing[1];
                let pz = vz / spacing[2];
                
                // Interpolate normal
                let inx = bar[0] * n0[0].into() + bar[1] * n1[0].into() + bar[2] * n2[0].into();
                let iny = bar[0] * n0[1].into() + bar[1] * n1[1].into() + bar[2] * n2[1].into();
                let inz = bar[0] * n0[2].into() + bar[1] * n1[2].into() + bar[2] * n2[2].into();
                
                // Normalize
                let inn = (inx * inx + iny * iny + inz * inz).sqrt();
                let inx = if inn > 1e-10 { inx / inn } else { 0.0 };
                let iny = if inn > 1e-10 { iny / inn } else { 0.0 };
                let inz = if inn > 1e-10 { inz / inn } else { 0.0 };
                
                // Sample volume data at this position
                let volume_val = trilinear_interpolate_internal(volume, px, py, pz);
                
                // Apply window/level
                let wl = window_level as f64;
                let ww = window_width as f64;
                let min_val = wl - ww / 2.0;
                let max_val = wl + ww / 2.0;
                
                let gv = if volume_val.into() <= min_val {
                    0.0
                } else if volume_val.into() >= max_val {
                    255.0
                } else {
                    ((volume_val.into() - min_val) / ww + 0.5) * 255.0
                };
                
                let gv_idx = (gv as usize).min(255);
                
                // Apply color from CLUT
                texture_image[[y, x, 0]] = clut[[gv_idx, 0]];
                texture_image[[y, x, 1]] = clut[[gv_idx, 1]];
                texture_image[[y, x, 2]] = clut[[gv_idx, 2]];
                
                // Calculate gradient for texture normals (simplified)
                let h = 1.0;
                let gx1 = trilinear_interpolate_internal(volume, px - h, py, pz);
                let gx2 = trilinear_interpolate_internal(volume, px + h, py, pz);
                let gy1 = trilinear_interpolate_internal(volume, px, py - h, pz);
                let gy2 = trilinear_interpolate_internal(volume, px, py + h, pz);
                let gz1 = trilinear_interpolate_internal(volume, px, py, pz - h);
                let gz2 = trilinear_interpolate_internal(volume, px, py, pz + h);
                
                let tnx = (gx2.into() - gx1.into()) / (2.0 * h);
                let tny = (gy2.into() - gy1.into()) / (2.0 * h);
                let tnz = (gz2.into() - gz1.into()) / (2.0 * h);
                
                let tnn = (tnx * tnx + tny * tny + tnz * tnz).sqrt();
                
                if tnn > 1e-10 {
                    texture_normals[[y, x, 0]] = (((tnx / tnn) + 1.0) * 127.5) as u8;
                    texture_normals[[y, x, 1]] = (((tny / tnn) + 1.0) * 127.5) as u8;
                    texture_normals[[y, x, 2]] = (((tnz / tnn) + 1.0) * 127.5) as u8;
                }
            }
        }
    }
    
    (tcoords, texture_image, texture_normals)
}
