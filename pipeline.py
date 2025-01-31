import subprocess
import papermill as pm
import os
import json


requirements_file = "requirements.txt"
# Paths
# notebooks_dir = "notebooks/"
output_dir = "output/"
output_der = "output/"
äinput_model_name = "apple/mobilevit-xx-small"  # default model is apple/mobilevit-xx-small
input_model_name = "tiny_vit_5m_224"

# model_path = "apple_mobilevit-xx-small_2025-01-12_17-23.pth" #default model for testing pipline
model_path = f"models/tiny_vit_5m_224_2025-01-06_17-38.pth"


# Function to install requirements
def install_requirements():
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.check_call(["pip", "install", "-r", requirements_file])
    else:
        print(f"Error: {requirements_file} not found. Please ensure the file exists.")
        exit(1)


# Install dependencies
# install_requirements()

notebooks = {
    "watermarking": os.path.join("creating_watermarked_images.ipynb"),
    "training": os.path.join("train_classifier_and_classify_with_and_without_waterma.ipynb"),
    "classify": os.path.join("classify_with_and_without_watermark.ipynb"),
    "removal": os.path.join("watermark_removal.ipynb"),
    "final_classify": os.path.join("classify_reconstruced_images.ipynb"),
    "xai": os.path.join("pytorch_grad_cam_explanation.ipynb"),
    "metrics": os.path.join("pytorch_grad_cam_explanation.ipynb"),
}

# Paths to save executed notebooks
executed_notebooks = {
    "watermarking": os.path.join(output_dir, "executed_watermarked.ipynb"),
    "training": os.path.join(output_dir, "executed_training.ipynb"),
    "classify": os.path.join("executed_classify.ipynb"),
    "removal": os.path.join(output_dir, "executed_removal.ipynb"),
    "final_classify": os.path.join(output_dir, "final_classify_notebook.ipynb"),
    "xai": os.path.join(output_dir, "executed_pytorch_grad_cam_explanation.ipynb"),
    "metrics": os.path.join(output_dir, "executed_metrics.ipynb"),
}

# Run notebooks sequentially
print("Starting pipeline...")

print("Creating watermarked images...")
# opacity parameters
# pm.execute_notebook(
#    notebooks["watermarking"],
#    executed_notebooks["watermarking"],
#    parameters={"output_dir": os.path.join(output_dir, "watermarked_images/")},
#    kernel_name="python3"
# )

"""
# Step 2: Train classifier and classify with/without watermark
# Load one pretrained model
#model_name = "apple/mobilevit-xx-small"
#model_name = "tiny_vit_5m_224"
print("Training classifier...")
print("Classifier: apple/mobilevit-xx-small")
#print("Classifier: tiny_vit_5m_224")
pm.execute_notebook(
    notebooks["training"],
    executed_notebooks["training"],
    parameters={"output_dir": output_dir,
               "input_model_name" : input_model_name}
)

#Read Saved Epoch Stats
print("Reading epoch stats from training output...")
epoch_stats_path = os.path.join(output_dir, "epoch_stats.json")
with open(epoch_stats_path, "r") as f:
    epoch_stats = json.load(f)

# Display stats in the pipeline
print("\nEpoch Stats from Training:")
for stat in epoch_stats:
    print(f"Epoch {stat['epoch']}: "
          f"Train Loss: {stat['train_loss']:.4f}, "
          f"Train Accuracy: {stat['train_accuracy']:.4f}, "
          f"Val Loss: {stat['val_loss']:.4f}, "
          f"Val Accuracy: {stat['val_accuracy']:.4f}, "
          f"F1 Score: {stat['f1_score']:.4f}")
    
   
    
# Step xyz: 
print("Classification of images with and without watermark...")
pm.execute_notebook(
    notebooks["classify"],
    executed_notebooks["classify"],
    parameters={
        "output_dir": output_dir,
        "output_der": output_der,       
        "input_model_name" : input_model_name,
        "input_model_path" : model_path
    }
)

#show final classifcation of reconstructed images
# Load the test metrics from the JSON file
metrics_path = os.path.join("output/test_metrics", "test_metrics.json")
with open(metrics_path, "r") as f:
    test_metrics = json.load(f)

# Print the metrics in the pipeline
print(f"Test Metrics: {test_metrics}")

metrics_path = os.path.join("output/prediction_metrics", "prediction_metrics.json")
with open(metrics_path, "r") as f:
    prediction_metrics = json.load(f)

# Print the metrics in the pipeline
print(f"Prediction Metrics: {prediction_metrics}")

"""

# Step 3: Remove watermarks
# print("Removing watermarks...")
# pm.execute_notebook(
#    notebooks["removal"],
#    executed_notebooks["removal"],
#    parameters={
#        "watermarked_dir": os.path.join(output_dir, "watermarked_images/"),
#        "cleaned_dir": os.path.join(output_dir, "cleaned_images/")
#    }
# )

# show some pictures
"""
# Step 4: 
#print("Classification of reconstructed images...")
pm.execute_notebook(
    notebooks["final_classify"],
    executed_notebooks["final_classify"],
    parameters={
        "output_dir": output_dir,        
        "input_model_name" : input_model_name,
        "input_model_path" : model_path
    }
)

#show final classifcation of reconstructed images
# Load the test metrics from the JSON file
metrics_path = os.path.join("output/reconstructed_metrics", "reconstructed_metrics.json")
with open(metrics_path, "r") as f:
    reconstructed_metrics = json.load(f)

# Print the metrics in the pipeline
print(f"Reconstructed Metrics: {reconstructed_metrics}")
"""

print("XAI evaluation...")
pm.execute_notebook(
    notebooks["xai"],
    executed_notebooks["xai"],
    parameters={
        "output_dir": output_dir,
        "input_model_path": model_path,
        "input_model_name": input_model_name,
    },
)

# Path to XAI outputs
xai_output_path = os.path.join(output_dir, "xai_outputs", "gradcam_output.png")

print(f"Grad-CAM output available at: {xai_output_path}")

# Optionally, display the image in an interactive session
# This requires an environment like Jupyter Notebook or JupyterLab
from IPython.display import Image, display

display(Image(filename=xai_output_path))

# show outputs etc.

print("Metrics evaluation...")
pm.execute_notebook(
    notebooks["metrics"],
    executed_notebooks["metrics"],
    parameters={
        "output_dir": output_dir,
        "input_model_path": model_path,
        "input_model_name": input_model_name,
    },
)

# Path to metrics outputs
metrics_output_path = os.path.join(output_dir, "metrics_outputs", "metrics.png")

print(f"Metric output available at: {metrics_output_path}")

# Optionally, display the image in an interactive session
# This requires an environment like Jupyter Notebook or JupyterLab
from IPython.display import Image, display

display(Image(filename=metrics_output_path))


print("Pipeline execution completed!")
