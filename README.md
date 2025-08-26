# Ray Tracing Scaffold Prediction

This project implements a ray-tracing algorithm to process 3D models in OBJ format and predict scaffold types using a pre-trained model.

## Project Structure

```
raytracing-scaffold-prediction
├── data_example
│   ├── CF_ex
│       ├── CF_ex.obj                  # Example 3D model in OBJ format
│       ├── distance_map_1.npy         # First distance map generated
│       ├── distance_map_1.png         # First distance map image for visualization
│       ├── distance_map_2.npy         # Second distance map generated
│       ├── distance_map_2.png         # Second distance map image for visualization
│       └── predictions.json           # Predictions of scaffold types
│   ├── CG_ex
│       ├ ... (same)
│   ├ ... (Same folders for FG, MF, MG, NF, NF+OS, PPS, SC and SC+OS)
│
├── models
│   └── model_inception_actin_plane_filtered.pth  # Pre-trained model for predictions
├── src
│   ├── process_obj.py                   # Script to process OBJ file, generate distance maps amd predict
│   ├── predict_scaffold.py              # Script to load model and predict scaffold types
│   └── utils
│       └── __init__.py                 # Utility functions for the project
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your OBJ file in the `data/sample_name` directory. The example OBJ file is provided as `sample_name.obj`, with sample_name to 'CF_ex', CG_ex', ... but it can have any name. You just need to have only one .obj file in this folder.
2. Go to the src/ folder and run the ray-tracing algorithm to generate distance maps:

```bash
python src/process_obj.py --samples_name 'sample_name' data_folder 'data'
```

3. After generating the distance maps, use the pre-trained model to predict the scaffold type:

```bash
python src/predict_scaffold.py
```

4. The predictions will be saved in `data/output/predictions.json`.

## Notes

- Ensure that the model file `model_inception_actin_plane_filtered.pth` is present in the `models` directory.
- The output distance maps will be saved in the `data/output` directory.