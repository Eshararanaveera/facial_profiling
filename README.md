# Grooming Component 1

Image classification models for identifying **jawline** and **hairline** types. The project uses PyTorch and EfficientNet-B0 models and provides a command-line prediction script that returns the predicted class, confidence score, and class probabilities.

## Features

- Jawline classification: `angular`, `sharp`, and `soft`
- Hairline classification: `m_shaped` and `straight`
- EfficientNet-B0 model architecture
- CPU and CUDA inference support
- JSON prediction output
- Configurable confidence threshold stored with each trained model

## Project Structure

```text
grooming_component_1/
|
├── README.md
├── requirements.txt
├── .gitignore
├── sample_hairline.jpg                 # Example hairline input
├── sample_jawline.jpg                  # Example jawline input
|
├── configs/                             # Project configuration files
|
├── data/                                # Local dataset and generated data
│   ├── raw/
│   │   ├── hairline/
│   │   │   ├── excluded/
│   │   │   ├── m_shaped/
│   │   │   └── straight/
│   │   ├── jawline/
│   │   │   ├── angular/
│   │   │   ├── excluded/
│   │   │   ├── sharp/
│   │   │   └── soft/
│   │   ├── labels/
│   │   │   ├── hairline_labels.csv
│   │   │   └── jawline_labels.csv
│   │   ├── processed/                   # Preprocessed images
│   │   └── splits/                       # Train/validation/test splits
│   │       ├── hairline/
│   │       └── jawline/
│   └── output/                          # Dataset processing output
|
├── models/
│   ├── hairline/
│   │   └── hairline_efficientnet_b0.pth
│   └── jawline/
│       └── jawline_efficientnet_b0.pth
|
├── notebooks/                           # Experiments and analysis notebooks
|
├── outputs/
│   ├── hairline/                        # Reports and predictions
│   └── jawline/
|
├── scripts/
│   ├── check_dataset.py
│   ├── check_images.py
│   ├── evaluate_model.py
│   ├── predict.py                       # Main inference script
│   ├── preprocess_faces.py
│   ├── split_dataset.py
│   ├── train_classifier.py
│   ├── train_hairline.py
│   └── train_jawline.py
|
└── venv/                                # Local Python virtual environment
```

> `data/`, `models/`, `outputs/`, and `venv/` are intentionally excluded from Git because they can contain large datasets, generated files, model weights, or local environment files. Keep the folder structure locally when setting up the project.

## Requirements

- Python 3.10 or newer
- PyTorch
- Torchvision
- Pillow

Create and activate a virtual environment from the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision pillow
```

If PowerShell blocks activation, run this once in the current PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Prediction

Run all commands from the `grooming_component_1` directory.

### Jawline

```powershell
python scripts/predict.py `
	--task jawline `
	--model models/jawline/jawline_efficientnet_b0.pth `
	--image sample_jawline.jpg `
	--output outputs/jawline/prediction.json
```

### Hairline

```powershell
python scripts/predict.py `
	--task hairline `
	--model models/hairline/hairline_efficientnet_b0.pth `
	--image sample_hairline.jpg `
	--output outputs/hairline/prediction.json
```

The `--image` value can be any valid image path. Relative paths are resolved from the directory where the command is executed.

Example output:

```json
{
	"task": "jawline",
	"model": "jawline_efficientnet_b0.pth",
	"image": "sample_jawline.jpg",
	"device": "cpu",
	"result": {
		"label": "soft",
		"confidence": 0.82,
		"confidence_percent": 82.0,
		"class_probabilities": {
			"angular": 0.08,
			"sharp": 0.10,
			"soft": 0.82
		},
		"message": "Prediction accepted."
	}
}
```

When the highest confidence is below the model threshold, the final label is returned as `uncertain` and the output includes a recommendation to provide a clearer frontal image.

## Dataset Organization

Place source images inside their matching class directories:

```text
data/raw/jawline/angular/
data/raw/jawline/sharp/
data/raw/jawline/soft/

data/raw/hairline/m_shaped/
data/raw/hairline/straight/
```

Keep excluded or low-quality images in the relevant `excluded/` directory. Annotation CSV files should record the image path, participant ID, quality checks, pose checks, and final label.

## Utility Scripts

| Script | Purpose |
| --- | --- |
| `check_dataset.py` | Validate dataset organization and labels |
| `check_images.py` | Check image files for readability |
| `preprocess_faces.py` | Prepare face images for the dataset |
| `split_dataset.py` | Create train, validation, and test splits |
| `train_classifier.py` | Shared classifier training logic |
| `train_hairline.py` | Hairline model training entry point |
| `train_jawline.py` | Jawline model training entry point |
| `evaluate_model.py` | Generate evaluation metrics and reports |
| `predict.py` | Run inference on one image |

## Git Notes

The repository ignores virtual environments, model weights, datasets, generated outputs, logs, and IDE files. Before sharing the project, include only source code, configuration, documentation, and small example assets that are safe to commit.

## License

Add the project license here before distributing the code or trained models.
