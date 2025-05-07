# Ray Tracing Scaffold Prediction

This project implements a ray-tracing algorithm to process 3D models in OBJ format and predict scaffold types using a pre-trained model.

## Project Structure

```
raytracing-scaffold-prediction
├── data
│   ├── sample_example_1
│       ├── example.obj                # Example 3D model in OBJ format
│       ├── distance_map_1.npy         # First distance map generated
│       ├── distance_map_2.npy         # Second distance map generated
│       └── predictions.json            # Predictions of scaffold types
├── models
│   └── model_inception_actin_plane_filtered.pth  # Pre-trained model for predictions
├── src
│   ├── process_obj.py                  # Script to process OBJ file and generate distance maps
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

1. Place your OBJ file in the `data/input` directory. The example OBJ file is provided as `example.obj`.
2. Run the ray-tracing algorithm to generate distance maps:

```bash
python src/process_obj.py
```

3. After generating the distance maps, use the pre-trained model to predict the scaffold type:

```bash
python src/predict_scaffold.py
```

4. The predictions will be saved in `data/output/predictions.json`.

## Notes

- Ensure that the model file `model_inception_actin_plane_filtered.pth` is present in the `models` directory.
- The output distance maps will be saved in the `data/output` directory.

## License

This project is licensed under the MIT License.