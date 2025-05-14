import argparse
import os, sys
from contextlib import contextmanager
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import plotly.graph_objects as go
from utils.helpers_plotting import plot_points, plot_surface
#from vmtk import vmtkscripts
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars
import torch

@contextmanager
def suppress_stdout():
    """Suppress stdout of VTK functions"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def create_mesh(path):
    """Create mesh from OBJ file

    Args:
        path (str): path to OBJ file

    Returns:
        vtkPolyData: mesh
    """
    with suppress_stdout():
        reader = vtk.vtkOBJReader()
        reader.SetFileName(path)
        reader.Update()

        # Smooth and fill holes
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetNumberOfIterations(300)
        smoothFilter.SetRelaxationFactor(0.1)
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.SetInputConnection(reader.GetOutputPort())
        smoothFilter.Update()

        fill = vtk.vtkFillHolesFilter()
        fill.SetInputConnection(smoothFilter.GetOutputPort())
        fill.SetHoleSize(1000000000)
        fill.Update()

        # Write STL file
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(path[:-4] + '.stl')
        writer.SetInputConnection(fill.GetOutputPort())
        writer.Write()

        return fill.GetOutput()

def read_mesh(path):
    """Read mesh from STL file
    
    Args:
        path (str): path to STL file
        
    Returns:
        vtkPolyData: mesh
    """
    with suppress_stdout():
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()


def fit_plane(mesh):
    """Fit a plane with GPU acceleration and memory management."""
    print("Computing mesh properties... (GPU mode)")
    
    # Get mesh points
    vtk_points = mesh.GetPoints().GetData()
    n_points = vtk_points.GetNumberOfTuples()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available, using GPU acceleration")
        device = 'cuda'
    else:
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Process in batches for large meshes
    batch_size = 500000
    if n_points > batch_size:
        # For huge meshes, sample points
        print(f"Sampling {batch_size} points from {n_points} total points")
        indices = np.random.choice(n_points, batch_size, replace=False)
        points = np.array([vtk_points.GetTuple(i) for i in indices])
    else:
        points = np.array([vtk_points.GetTuple(i) for i in range(n_points)])
        
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    # Clear memory before GPU operations
    del points
    import gc
    gc.collect()
    
    # Perform SVD on GPU with careful memory handling
    try:
        # Move data to GPU
        centered_tensor = torch.tensor(centered, device=device, dtype=torch.float32)
        
        # Clear NumPy array to free CPU memory
        del centered
        gc.collect()
        
        # Make sure there's enough GPU memory
        torch.cuda.empty_cache()
        
        # Perform SVD
        _, _, vh = torch.svd(centered_tensor)
        
        # Move result back to CPU immediately
        directionx = vh[:, 0].cpu().numpy()
        directiony = vh[:, 1].cpu().numpy()
        normal = vh[:, 2].cpu().numpy()
        
        # Clear GPU memory
        del centered_tensor, vh
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        # If GPU memory error, fallback to CPU incremental method
        print(f"GPU error: {e}")
        print("Falling back to incremental CPU method")
        
        # Free any GPU memory
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        # Use the incremental approach from Option 1
        return fit_plane_incremental(mesh)
    
    print("Plane fitting completed")
    return directionx, directiony, normal, centroid

# Add this helper function for fallback
def fit_plane_incremental(mesh):
    """Memory-efficient plane fitting using incremental covariance computation."""
    # [Copy the code from Option 1]
    # ...

def compute_top_bottom_points(center, normal):
    """Compute the top and bottom points normal to the plane.
    
    Args:
    center (np.array): Center of the mesh.
    normal (np.array): Normal direction of the plane.
    distance (float): Distance to move along the normal direction.
    
    Returns:
    top_point (np.array): Top point normal to the plane.
    bottom_point (np.array): Bottom point normal to the plane.
    """
    distance = 15
    top_point = center + normal * distance
    bottom_point = center - normal * distance
    return top_point, bottom_point

def create_transform(center, directionx, directiony, normal, inverse=False):
    """Create transform matrix from PCA directions."""
    transform = vtk.vtkTransform()
    directionz = np.cross(directionx, directiony)
    
    if inverse:
        directionx = -directionx
        directionz = np.cross(directionx, directiony)
    
    T = np.eye(4)
    T[:3, 0] = directionx
    T[:3, 1] = directiony
    T[:3, 2] = directionz
    T[:3, 3] = center
    transform.SetMatrix(T.flatten())
    return transform


def check_hemisphere_orientation(points, reference_point, center, normal):
    """Check if sampled points are on the same side of the plane as the reference point.
    
    Args:
        points (np.array): Sampled hemisphere points
        reference_point (np.array): Reference point (top_point or bottom_point)
        center (np.array): Center of the plane
        normal (np.array): Normal vector of the plane
        
    Returns:
        bool: True if points are on the same side as reference_point, False otherwise
    """
    print("Checking hemisphere orientation...")
    # Vector from center to reference point
    center_to_ref = reference_point - center
    
    # Check if reference point is above or below plane using dot product with normal
    ref_side = np.dot(center_to_ref, normal) > 0
    
    # Sample some points to check (using first 10 points for efficiency)
    sample_count = min(10, len(points))
    sampled_points = points[:sample_count]
    
    # Check which side of the plane each sampled point is on
    correct_side_count = 0
    for point in sampled_points:
        center_to_point = point - center
        point_side = np.dot(center_to_point, normal) > 0
        if point_side == ref_side:
            correct_side_count += 1
    
    # If majority of points are on the correct side, return True
    return correct_side_count > sample_count / 2


def sample_hemisphere(density, transform, radius=1.0):
    """Sample points on a hemisphere using a given transform matrix."""
    points = []
    for i in np.linspace(0, np.pi/2, density):
        for j in np.linspace(0, 2*np.pi, density):
            x = radius * np.sin(i) * np.cos(j)
            y = radius * np.sin(i) * np.sin(j)
            z = radius * np.cos(i)
            p = np.array([x, y, z])
            points.append(transform.TransformPoint(p))
    return np.array(points)

def sample_hemisphere_relative_to_plane(mesh, transform, density, above_plane=True, scale_factor=1.2):  # Increased scale_factor
    """Sample a hemisphere oriented relative to the best-fit plane of the mesh."""
    bounds = mesh.GetBounds()
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    radius = scale_factor * max_dim  # Increased radius 
    points = sample_hemisphere(density, transform, radius)
    return points

def distance_mapping(surface, ray_source, ray_targets, n_jobs=8, single_batch=False):
    """Compute distance mapping from ray source to ray targets on surface

    Args:
        surface (vtkPolyData): mesh
        ray_source (np.array): source point
        ray_targets (np.array): target points
        n_jobs (int): number of parallel jobs (used as fallback)
        single_batch (bool): if True, process as a single batch (no progress bar)

    Returns:
        distances (np.array): distance mapping
    """
    # Set up OBB tree for ray casting
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(surface)
    obbTree.BuildLocator()
    
    # Get total number of points to process
    total_points = ray_targets.shape[0]
    
    # Start timing
    import time
    start_time = time.time()
    
    # If single_batch=True, process directly without progress bar and threading
    if single_batch:
        # Process the entire batch directly
        distances = np.zeros(total_points).astype(float)
        intersection_count = 0
        
        # Add a mini progress indicator for large batches
        for i in range(total_points):
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                est_total = (elapsed / i) * total_points
                remaining = est_total - elapsed
                print(f"  Progress: {i}/{total_points} points ({i/total_points*100:.1f}%), " +
                      f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
            
            distance = find_intersect_on_surface(obbTree, ray_source, ray_targets[i])
            distances[i] = distance
            if distance > 0:
                intersection_count += 1
        
        # Report batch statistics
        elapsed_time = time.time() - start_time
        intersection_rate = intersection_count / total_points if total_points > 0 else 0
        print(f"Batch completed in {elapsed_time:.2f} seconds")
        print(f"Intersection rate: {intersection_rate:.2%} ({intersection_count}/{total_points} rays hit surface)")
        
        return distances
    
    # Otherwise, use parallel processing with progress bar
    pbar = tqdm(total=total_points, desc="Ray casting progress")

    # Use parallel processing with progress tracking
    pool = Pool()
    size = int(np.ceil(ray_targets.shape[0] / n_jobs))
    parallel_func = partial(find_intersect_parallel, obbTree, ray_source, ray_targets, size, pbar)
    outputs_async = pool.map(parallel_func, range(n_jobs))
    pool.close()
    pool.join()
    
    # Close the progress bar
    pbar.close()
    
    # Combine results
    distances = np.zeros(ray_targets.shape[0]).astype(float)
    total_intersections = 0
    for i in range(n_jobs - 1):
        if i * size < len(distances):
            end_idx = min((i + 1) * size, len(distances))
            distances[i * size: end_idx] = outputs_async[i][0][:end_idx-i*size]
            total_intersections += outputs_async[i][1]
    
    if (n_jobs - 1) * size < len(distances):
        distances[(n_jobs - 1) * size:] = outputs_async[n_jobs - 1][0][:len(distances)-(n_jobs-1)*size]
        total_intersections += outputs_async[n_jobs - 1][1]
    
    # Report overall statistics
    elapsed_time = time.time() - start_time
    intersection_rate = total_intersections / total_points if total_points > 0 else 0
    print(f"Full distance mapping completed in {elapsed_time:.2f} seconds")
    print(f"Overall intersection rate: {intersection_rate:.2%} ({total_intersections}/{total_points} rays hit surface)")
    
    return distances

def find_intersect_parallel(obbTree, ray_source_point, ray_target_points, size, pbar, job):
    """Find intersection points on surface using OBBTree (parallel version) with progress tracking

    Args:
        obbTree (vtkOBBTree): OBBTree locator
        ray_source_point (np.array): source point
        ray_target_points (np.array): target points
        size (int): size of each job
        pbar (tqdm): progress bar
        job (int): job index
        
    Returns:
        distances (np.array): distances
    """
    start = size * job
    end = min(size * (job + 1), ray_target_points.shape[0])
    
    distances = np.zeros(end - start).astype(float)
    intersection_count = 0
    
    # Process each ray
    for i in range(start, end):
        if i < ray_target_points.shape[0]:  # Safety check
            distance = find_intersect_on_surface(obbTree, ray_source_point, ray_target_points[i])
            distances[i - start] = distance
            if distance > 0:
                intersection_count += 1
        
        # Update progress bar (thread-safe)
        with pbar.get_lock():
            pbar.update(1)
    
    # Return both distances and intersection statistics
    intersection_rate = intersection_count / (end - start) if (end - start) > 0 else 0
    return distances, intersection_count, intersection_rate

def find_intersect_on_surface(surface_tree, ray_source_pt, ray_target_pt):
    """Find all intersection points and return maximum distance"""
    # Calculate the direction from source to target
    direction = ray_target_pt - ray_source_pt
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm > 1e-6:  # Ensure non-zero length
        # Normalize the direction
        direction = direction / direction_norm
        
        # Cast a VERY long ray in this direction
        ray_end = ray_source_pt + direction * 1000000  # Increased length
        
        # Perform ray casting to find intersection points
        intersection_pts = ray_casting(surface_tree, ray_source_pt, ray_end)
        
        if intersection_pts and len(intersection_pts) > 0:
            # Calculate distances from source to each intersection point
            distances = [np.linalg.norm(np.array(ray_source_pt) - np.array(point)) for point in intersection_pts]
            
            # Return maximum distance
            if distances:
                return max(distances)
    
    # Return 0 if no intersections found or other issues
    return 0.0

def ray_casting(surface_tree, pt0, pt1):
    """Ray casting on surface using OBBTree, returns all intersection points

    Args:
        surface_tree (vtkOBBTree): OBBTree
        pt0 (np.array): source point
        pt1 (np.array): target point

    Returns:
        points (list): list of intersection points
    """
    # Ensure points are numpy arrays
    pt0 = np.array(pt0, dtype=float)
    pt1 = np.array(pt1, dtype=float)
    
    # Check for invalid values
    if not np.all(np.isfinite(pt0)) or not np.all(np.isfinite(pt1)):
        print("Warning: Invalid ray points (NaN or Inf)")
        return None
    
    # Check if points are too close to each other (would cause precision issues)
    if np.linalg.norm(pt1 - pt0) < 1e-6:
        print("Warning: Source and target points are too close")
        return None

    # Initialize VTK points and cell IDs
    intersection_pts = vtk.vtkPoints()
    cell_ids = vtk.vtkIdList()
    
    # Add tolerance parameter to help with precision issues
    tolerance = 1e-6
    
    # Get all intersections - try with tolerance if available in your VTK version
    try:
        code = surface_tree.IntersectWithLine(pt0, pt1, tolerance, intersection_pts, cell_ids)
    except TypeError:
        # Fall back to standard method if tolerance parameter isn't supported
        code = surface_tree.IntersectWithLine(pt0, pt1, intersection_pts, cell_ids)
    
    # Get number of intersections even if code == 0
    n_points = intersection_pts.GetNumberOfPoints()
    
    if n_points > 0:
        points = []
        # Collect all intersection points
        for i in range(n_points):
            point = intersection_pts.GetPoint(i)
            points.append(np.array(point))
        return points
    else:
        #print("Warning: no intersection found in ray casting")
        return None

'''def plot_mesh_vmtk(surface, surface_color='lightblue'):
    """Plot mesh using plotly

    Args:
        surface (vtkPolyData): mesh
        surface_color (str): color of mesh

    Returns:
        fig (plotly.graph_objects.Figure): plotly figure
    """
    with suppress_stdout():
        numpy_converter = vmtkscripts.vmtkSurfaceToNumpy()
        numpy_converter.Surface = surface
        numpy_converter.Execute()
    return plot_surface(numpy_converter.ArrayDict['Points'], numpy_converter.ArrayDict['CellData']['CellPointIds'],
                        surface_color)
'''

def plot_mesh(surface, surface_color='lightblue'):
    """Plot mesh using plotly without VMTK."""
    # Convert VTK points to NumPy
    vtk_points = surface.GetPoints().GetData()
    points = vtk_to_numpy(vtk_points)

    # Convert cell connectivity
    vtk_cells = surface.GetPolys().GetData()
    cells_flat = vtk_to_numpy(vtk_cells)

    # Extract individual triangles from [3, id1, id2, id3, 3, id4, id5, id6, ...]
    cells = []
    i = 0
    while i < len(cells_flat):
        n = cells_flat[i]
        if n == 3:
            cells.append(cells_flat[i+1:i+4])
        i += n + 1

    cells = np.array(cells)

    return plot_surface(points, cells, surface_color)


def plot(path, surface, center, directionx, directiony, top_point, bottom_point, surface_color='lightblue', show_plot=False, save_plot=True):
    """Plot mesh and points using plotly
    
    Args:
        path (str): path to input STL or OBJ file
        surface (vtkPolyData): mesh
        center (np.array): center of mesh
        direction1 (np.array): PCA direction 1
        direction2 (np.array): PCA direction 2
        surface_color (str): color of mesh
        show_plot (bool): show plot
        save_plot (bool): save plot
    """
    key_points = [center, center + directionx * 10, center + directiony * 10, top_point, bottom_point]
    labels = ['center', 'X', 'Y', 'Top', 'Bottom']
    colors = ['red', 'blue', 'purple', 'green', 'orange']
    points_plot = plot_points(key_points, labels, colors)

    mesh_plot = plot_mesh(surface, surface_color)

    fig = go.Figure(data=points_plot.data + mesh_plot.data)

    fig.update_layout(width=500, height=600,
                        scene=dict(
                            xaxis=dict(showticklabels=False, title='sagittal', backgroundcolor="rgba(0, 0, 0,0)",
                                        gridcolor='lightgray'),
                            yaxis=dict(showticklabels=False, title='coronal', backgroundcolor="rgba(0, 0, 0,0)",
                                        gridcolor='lightgray'),
                            zaxis=dict(showticklabels=False, title='axial', backgroundcolor="rgba(0, 0, 0,0)",
                                        gridcolor='lightgray'), ),
                        margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(scene_camera=dict(eye=dict(x=2, y=2, z=0.75)), scene_aspectmode='data')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.9,
        xanchor="right",
        x=0.85
    ))
    if show_plot:
        fig.show()
    if save_plot:
        fig.write_image(path[:-4] + '_img.jpg')


def debug_plot(surface, top_points, bottom_points, point_top, point_bottom, normal, center, directionx, directiony, show_plot=True, save_plot=False, path=None, ray_fraction=0.1):
    """Create a debug plot to visualize hemispheres and rays with all intersections.
    
    Args:
        surface (vtkPolyData): The input mesh surface
        top_points (np.array): Points of the top hemisphere
        bottom_points (np.array): Points of the bottom hemisphere
        point_top (np.array): Top ray source point
        point_bottom (np.array): Bottom ray source point
        normal (np.array): Normal vector of the fitted plane
        center (np.array): Center point
        directionx (np.array): X direction of the fitted plane
        directiony (np.array): Y direction of the fitted plane
        show_plot (bool): Whether to show the plot
        save_plot (bool): Whether to save the plot
        path (str): Path to save the plot image
        ray_fraction (float): Fraction of rays to display
    """

    traces = []

    # Create mesh plot
    mesh_plot = plot_mesh(surface, surface_color='lightblue')
    traces.extend(mesh_plot.data)

    # Add center, bottom and top points
    top_point_plot = go.Scatter3d(x=[point_top[0]], y=[point_top[1]], z=[point_top[2]], mode='markers', marker=dict(color='green', size=5), name='Top Point')
    bottom_point_plot = go.Scatter3d(x=[point_bottom[0]], y=[point_bottom[1]], z=[point_bottom[2]], mode='markers', marker=dict(color='red', size=5), name='Bottom Point')
    center_point_plot = go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode='markers', marker=dict(color='purple', size=5), name='Center Point')
    traces.extend([top_point_plot, bottom_point_plot, center_point_plot])

    # Add normal vector
    normal_length = 3
    normal_vector_center = go.Scatter3d(x=[center[0], center[0] + normal[0] * normal_length], y=[center[1], center[1] + normal[1] * normal_length], z=[center[2], center[2] + normal[2] * normal_length], mode='lines', line=dict(color='purple', width=3), name='Normal Vector')
    traces.extend([normal_vector_center])
    
    # Add plane visualization
    plane_size = 75  # Adjust the size of the plane as needed
    plane_points = np.array([
        center + directionx * plane_size + directiony * plane_size,
        center + directionx * plane_size - directiony * plane_size,
        center - directionx * plane_size - directiony * plane_size,
        center - directionx * plane_size + directiony * plane_size
    ])
    plane_plot = go.Mesh3d(x=plane_points[:, 0], y=plane_points[:, 1], z=plane_points[:, 2], color='yellow', opacity=0.3, name='Fitted Plane')
    traces.extend([plane_plot])
    
    # Add hemisphere points
    top_scatter = go.Scatter3d(
        x=top_points[:, 0], y=top_points[:, 1], z=top_points[:, 2],
        mode='markers', marker=dict(color='green', size=2, opacity=0.3),
        name='Top Hemisphere'
    )
    
    bottom_scatter = go.Scatter3d(
        x=bottom_points[:, 0], y=bottom_points[:, 1], z=bottom_points[:, 2],
        mode='markers', marker=dict(color='red', size=2, opacity=0.3),
        name='Bottom Hemisphere'
    )
    traces.extend([top_scatter, bottom_scatter])
    
    fig = go.Figure(data=traces)
    
    # Update layout
    fig.update_layout(
        title="Hemisphere and Ray Debug Visualization",
        width=1000, height=800,
        scene=dict(
            xaxis=dict(showticklabels=True, title='X', backgroundcolor="rgba(0, 0, 0,0)", gridcolor='lightgray'),
            yaxis=dict(showticklabels=True, title='Y', backgroundcolor="rgba(0, 0, 0,0)", gridcolor='lightgray'),
            zaxis=dict(showticklabels=True, title='Z', backgroundcolor="rgba(0, 0, 0,0)", gridcolor='lightgray')
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        annotations=[
            dict(
                x=0.0,
                y=1.05,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=14),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
        ]
    )
    
    if show_plot:
        fig.show()
    if save_plot and path is not None:
        fig.write_image(path)

def main(path, density, mode = "stl"):
    """Main function to compute 2D surface distance mapping
    
    Args:
        path (str): path to input STL or OBJ file
        density (int): number of sampling angles in each dimension
        mode (str): mode of input file (obj or stl)
    """
    surface = None
    if mode == "obj":
        surface = create_mesh(path)
    elif mode == "stl":
        surface = read_mesh(path)
        if (surface == None):
            print("We have a problem it's empty")
    else:
        raise ValueError("Error: mode should be either obj or stl")

    print("Computing mesh properties...")
    directionx, directiony, normal, centroid = fit_plane(surface)
    top_point, bottom_point = compute_top_bottom_points(centroid, normal)

    print("Sampling points...")
    # First attempt with default inverse values
    inverse_default = False
    transform_top = create_transform(centroid, directionx, directiony, normal, inverse=inverse_default)
    points_top = sample_hemisphere_relative_to_plane(surface, transform_top, density, above_plane=True)
    
    # Check if points_top are on the same side as top_point
    top_correct = check_hemisphere_orientation(points_top, top_point, centroid, normal)
    if not top_correct:
        print("Correcting hemisphere orientation...")
        inverse_corrected = not inverse_default
        transform_top = create_transform(centroid, directionx, directiony, normal, inverse=inverse_corrected)
        points_top = sample_hemisphere_relative_to_plane(surface, transform_top, density, above_plane=True)
    
        transform_down = create_transform(centroid, directionx, directiony, normal, inverse=not inverse_corrected)
        points_bottom = sample_hemisphere_relative_to_plane(surface, transform_down, density, above_plane=False)
    else:
        transform_down = create_transform(centroid, directionx, directiony, normal, inverse=not inverse_default)
        points_bottom = sample_hemisphere_relative_to_plane(surface, transform_down, density, above_plane=False)
    
    '''
    print('\n Debug plotting...')
    debug_plot(
        surface,
        points_top,
        points_bottom,
        top_point, 
        bottom_point, 
        normal,
        centroid,
        directionx,
        directiony,  
        show_plot=True, 
        save_plot=True, 
        path='debug_plot.png', 
        ray_fraction=0.1  # Note: parameter is ray_fraction, not display_fraction
    )'''
    
    plot(path, surface, centroid, directionx, directiony, top_point, bottom_point)

    print("Computing distances...")
    print("Processing top distance map...")
    distances_top = distance_mapping(surface, top_point, points_bottom)
    
    print("Processing bottom distance map...")
    distances_bottom = distance_mapping(surface, bottom_point, points_top)
    
    print("Creating distance maps and saving results...")
    img1 = distances_top.reshape((density, density))
    np.save(path[:-4] + "_1.npy", img1)
    plt.figure()
    plt.imshow(img1)
    plt.colorbar()
    plt.savefig(path[:-4] + "_1.png")
    
    img2 = distances_bottom.reshape((density, density))
    np.save(path[:-4] + "_2.npy", img2)
    plt.figure()
    plt.imshow(img2)
    plt.colorbar()
    plt.savefig(path[:-4] + "_2.png")
    
    print("Distance mapping complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute a 2D surface distance mapping of a given 3D volume in OBJ format", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-po", "--path_obj", default=None, help="path to OBJ file containing 3D volume")
    parser.add_argument("-ps", "--path_stl", default=None, help="path to STL file containing 3D volume")
    parser.add_argument("-d", "--density", type = int, default = 1024, help="number of sampling angles in each dimension")
    args = parser.parse_args()
    if args.path_obj is not None:
        main(args.path_obj, args.density, mode = "obj")
    elif args.path_stl is not None:
        main(args.path_stl, args.density, mode = "stl")
    else:
        print("Error: please provide a path to the OBJ or STL file containing the 3D volume")