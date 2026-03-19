// Python bindings for surface texture generation
// Provides Python-accessible functions for texture generation

use numpy::ToPyArray;
use pyo3::prelude::*;

use crate::texture::generate_surface_texture_internal;
use crate::types::{VertexArray, FaceArray, ImageTypes3};

/// Generate surface texture from mesh and volume data
///
/// This function creates a texture atlas from a 3D mesh by:
/// 1. Laying out all triangles in a 2D grid
/// 2. Generating UV coordinates for each triangle
/// 3. Sampling volume data at each UV position
/// 4. Applying window/level and color lookup table
///
/// Arguments:
/// * vertices - Mesh vertices (N x 3) - float64
/// * normals - Vertex normals (N x 3) - float64
/// * faces - Face indices (M x 3) - int32
/// * volume - 3D volume data - int16
/// * spacing - Volume spacing (3 elements) - float64
/// * window_width - Window width for visualization
/// * window_level - Window level for visualization
/// * clut - Color lookup table (256 x 3) - uint8
/// * texture_dim - Texture atlas dimension (default 5000)
///
/// Returns tuple of (tcoords, texture_image, texture_normals)
/// - tcoords: float64 array (M, 6)
/// - texture_image: uint8 array (dim, dim, 3)
/// - texture_normals: uint8 array (dim, dim, 3)
#[pyfunction]
pub fn generate_surface_texture(
    py: Python<'_>,
    vertices: VertexArray<'_>,
    normals: VertexArray<'_>,
    faces: FaceArray<'_>,
    volume: ImageTypes3<'_>,
    spacing: numpy::PyReadonlyArray1<'_, f64>,
    window_width: i32,
    window_level: i32,
    clut: numpy::PyReadonlyArray2<'_, u8>,
    texture_dim: usize,
) -> PyResult<Py<pyo3::PyAny>> {
    
    let spacing_arr = spacing.as_array();
    let spacing_slice = [spacing_arr[0], spacing_arr[1], spacing_arr[2]];
    
    // Handle different input types - only support the main types used by Cython
    match (vertices, normals, faces, volume, clut) {
        (
            VertexArray::F64(vertices),
            VertexArray::F64(normals),
            FaceArray::I32(faces),
            ImageTypes3::I16(volume),
            clut,
        ) => {
            // Call with explicit type parameters: <f64, i32, i16>
            let (tcoords, texture_image, texture_normals) = generate_surface_texture_internal::<f64, i32, i16>(
                vertices.as_array().view(),
                normals.as_array().view(),
                faces.as_array().view(),
                volume.as_array().view(),
                &spacing_slice,
                window_width,
                window_level,
                clut.as_array().view(),
                texture_dim,
            );
            
            // Convert each array to Python using to_pyarray, then to PyObject
            let tcoords_py: Py<pyo3::PyAny> = tcoords.to_pyarray(py).into();
            let texture_image_py: Py<pyo3::PyAny> = texture_image.to_pyarray(py).into();
            let texture_normals_py: Py<pyo3::PyAny> = texture_normals.to_pyarray(py).into();
            
            // Return as tuple
            let result = pyo3::types::PyTuple::new(py, &[tcoords_py, texture_image_py, texture_normals_py])?;
            Ok(result.into())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported input types. Use: float64 vertices/normals, int32 faces, int16 volume"
        ))
    }
}
