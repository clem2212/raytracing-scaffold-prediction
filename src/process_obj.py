from pathlib import Path
import numpy as np
import torch
import os
from utils.ray_tracing_mapping_cells import create_mesh, fit_plane, sample_hemisphere_relative_to_plane, distance_mapping, compute_top_bottom_points, create_transform, check_hemisphere_orientation
import argparse
import json
import torch.nn as nn
from torchvision.models import inception_v3
import matplotlib.pyplot as plt

class InceptionScaffoldClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(InceptionScaffoldClassifier, self).__init__()
        
        # Load pretrained Inception v3
        self.inception = inception_v3(pretrained=True)        
        # Override the transform input method to handle 2 channels
        def new_transform_input(x):
            # Pad the 2-channel input to 3 channels
            x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
            if x.min() >= 0 and x.max() <= 1:
                x = x * 2 - 1
            return x
        
        self.inception._transform_input = new_transform_input
        
        # Modify first conv layer to accept 2 channels
        original_conv = self.inception.Conv2d_1a_3x3.conv
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(
            3, 32,  # Keep 3 input channels since we pad
            kernel_size=3,
            stride=2,
            bias=False
        )
        
        # Initialize the new conv layer
        with torch.no_grad():
            self.inception.Conv2d_1a_3x3.conv.weight.data[:, :2] = \
                original_conv.weight.data[:, :2]
            self.inception.Conv2d_1a_3x3.conv.weight.data[:, 2] = 0  # Zero-initialize the third channel
        
        # Modify the final classifier
        self.inception.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Disable auxiliary classifier during training
        self.inception.aux_logits = False

    def forward(self, x):
        return self.inception(x)


def process_obj_file(obj_file_path, output_dir, density=512, batch_size=10000):
    """Process the OBJ file to generate distance maps and save them as .npy and .png."""
    import time
    overall_start_time = time.time()
    
    print(f"Processing OBJ file: {obj_file_path}")
    # Load the mesh from the OBJ file
    try:
        surface = create_mesh(obj_file_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        raise

    # Compute mesh properties
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

    # Compute distance maps using batching
    print(f"Computing distance maps with density {density} and batch size {batch_size}...")
    print(f"Total rays to cast: {density * density}")
    
    # Prepare output arrays
    distances_top = np.zeros(density * density)
    distances_bottom = np.zeros(density * density)
    
    # Process in batches to save memory
    total_points = density * density
    batch_start_time = time.time()
    
    # Top distance map
    print("\nProcessing TOP distance map:")
    for i in range(0, total_points, batch_size):
        end_idx = min(i + batch_size, total_points)
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_points-1)//batch_size + 1} ({i}/{total_points})...")
        
        # Process batch for top distances
        batch_points = points_bottom[i:end_idx]
        batch_start = time.time()
        distances_top[i:end_idx] = distance_mapping(surface, top_point, batch_points, single_batch=True)
        print(f"Batch {i//batch_size + 1} completed in {time.time() - batch_start:.2f} seconds")
    
    top_map_time = time.time() - batch_start_time
    print(f"\nTop distance map completed in {top_map_time:.2f} seconds")
    
    # Bottom distance map
    print("\nProcessing BOTTOM distance map:")
    batch_start_time = time.time()
    for i in range(0, total_points, batch_size):
        end_idx = min(i + batch_size, total_points)
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_points-1)//batch_size + 1} ({i}/{total_points})...")
        
        # Process batch for bottom distances
        batch_points = points_top[i:end_idx]
        batch_start = time.time()
        distances_bottom[i:end_idx] = distance_mapping(surface, bottom_point, batch_points, single_batch=True)
        print(f"Batch {i//batch_size + 1} completed in {time.time() - batch_start:.2f} seconds")
    
    bottom_map_time = time.time() - batch_start_time
    print(f"\nBottom distance map completed in {bottom_map_time:.2f} seconds")

    # Save distance maps as .npy
    os.makedirs(output_dir, exist_ok=True)
    distance_map_1_path = Path(output_dir) / f"distance_map_1_{density}.npy"
    distance_map_2_path = Path(output_dir) / f"distance_map_2_{density}.npy"
    np.save(distance_map_1_path, distances_top)
    np.save(distance_map_2_path, distances_bottom)

    # Save distance maps as .png heatmaps
    distance_map_1_png_path = Path(output_dir) / f"distance_map_1_{density}.png"
    distance_map_2_png_path = Path(output_dir) / f"distance_map_2_{density}.png"

    plt.figure()
    plt.imshow(distances_top.reshape((density, density)), cmap="viridis")
    plt.colorbar()
    plt.title(f"Distance Map 1 (Top) - {density}x{density}")
    plt.savefig(distance_map_1_png_path)
    plt.close()

    plt.figure()
    plt.imshow(distances_bottom.reshape((density, density)), cmap="viridis")
    plt.colorbar()
    plt.title(f"Distance Map 2 (Bottom) - {density}x{density}")
    plt.savefig(distance_map_2_png_path)
    plt.close()

    total_time = time.time() - overall_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Distance maps saved to '{output_dir}' as .npy and .png.")

def load_model(model_path, num_classes=4):
    print(f"Loading model from {model_path}")
    model = InceptionScaffoldClassifier(num_classes=num_classes)
    # Load on CPU to avoid CUDA device mismatch issues
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_scaffold_type(model, distance_map_1, distance_map_2):
    """Predict the scaffold type from distance maps."""
    # Reshape the distance maps to 2D arrays
    distance_map_1 = distance_map_1.reshape(1024, 1024)  
    distance_map_2 = distance_map_2.reshape(1024, 1024)
    
    # Stack to create a 2-channel input
    combined_map = np.stack([distance_map_1, distance_map_2], axis=0)
    combined_map = torch.FloatTensor(combined_map)
    
    # Apply the same normalization as in training
    combined_map = (combined_map - combined_map.mean()) / (combined_map.std() + 1e-6)
    
    # Resize to match Inception input size (299×299)
    combined_map = torch.nn.functional.interpolate(
        combined_map.unsqueeze(0),  # Add batch dimension for interpolation
        size=(299, 299),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension
    
    # Convert to tensor and add batch dimension for model input
    input_tensor = combined_map.unsqueeze(0)  # Shape: [1, 2, 299, 299]
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        # Print raw outputs for debugging
        print(f"Raw model outputs: {outputs}")
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return predicted_class

def get_scaffold_name(class_idx):
    scaffold_names = {
        0: "Flat Surface",
        1: "Fibrous Scaffold",
        2: "Porous Sponge",
        3: "Hydrogel"
    }
    return scaffold_names.get(class_idx, f"Unknown ({class_idx})")

def save_predictions(predictions, output_path):
    scaffold_name = get_scaffold_name(predictions)
    result = {
        "predicted_class": predictions,
        "scaffold_name": scaffold_name
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"Predictions saved to {output_path}")
    print(f"Predicted scaffold type: {scaffold_name} (class {predictions})")

def main():
    parser = argparse.ArgumentParser(description="Process OBJ file and predict scaffold type.")
    parser.add_argument("--sample_name", required=True, help="Name of the sample to process.")
    parser.add_argument("--model_path", default="../models/model_inception_actin_plane_filtered.pth", 
                        help="Path to the pre-trained model.")
    parser.add_argument("--density", type=int, default=512, help="Density for sampling points.")
    parser.add_argument("--batch_size", type=int, default=10000, 
                        help="Batch size for ray casting to reduce memory usage.")
    parser.add_argument("--force_recompute", action="store_true", 
                        help="Force recomputation of distance maps even if they exist.")
    args = parser.parse_args()
    
    sample_name = args.sample_name
    model_path = args.model_path
    density = args.density
    batch_size = args.batch_size

    # STEP 1: Process the OBJ file to create distance maps
    print(f"Processing OBJ file for sample '{sample_name}'...")
    obj_file_path = f"../data/{sample_name}/{sample_name}.obj"
    output_dir = f"../data/{sample_name}/"
    
    # Step 2: Check if distance maps already exist
    distance_map_1_path = Path(output_dir) / f"distance_map_1_{density}.npy"
    distance_map_2_path = Path(output_dir) / f"distance_map_2_{density}.npy"

    if args.force_recompute or not distance_map_1_path.exists() or not distance_map_2_path.exists():
        print("Distance maps not found or recomputation forced. Computing distance maps...")
        try:
            process_obj_file(obj_file_path, output_dir, density, batch_size)
        except Exception as e:
            print(f"Error processing OBJ file: {e}")
            return
    else:
        print("Distance maps already exist. Skipping computation.")

    # STEP 3: Load the model and distance maps
    try:
        model = load_model(model_path)

        # Load the generated distance maps
        distance_map_1 = np.load(distance_map_1_path)
        distance_map_2 = np.load(distance_map_2_path)

        # Step 4: Predict the scaffold type
        predicted_class = predict_scaffold_type(model, distance_map_1, distance_map_2)

        # Step 5: Save the predictions
        predictions_path = Path(output_dir) / "predictions.json"
        save_predictions(predicted_class, predictions_path)
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()