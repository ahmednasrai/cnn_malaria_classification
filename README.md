# Malaria Cell Classification with a CNN

A deep learning project that classifies microscopic blood-cell images as **Parasitized** (infected with malaria) or **Uninfected**.

The analysis is in `Malaria_Cell_Classification.ipynb`. The model is a small Convolutional Neural Network built with TensorFlow/Keras.

## Dataset

The notebook downloads the **Cell Images for Detecting Malaria** dataset from Kaggle with `kagglehub` (`iarunava/cell-images-for-detecting-malaria`).

- Two balanced classes: `Parasitized` and `Uninfected`
- **27,558 images** in total (13,779 per class)
- Images have different sizes, so all are resized to **128 × 128** pixels before training

### Split

Each class is shuffled with a fixed seed and split **80% train / 20% test**:

| Split | Images per class | Total |
| --- | ---: | ---: |
| Train (80%) | 11,023 | 22,046 |
| Test (20%) | 2,756 | 5,512 |

A further **20% of the training folder** is used for validation, which gives the generators:

| Generator | Images |
| --- | ---: |
| Training (augmented) | 17,638 |
| Validation | 4,408 |
| Test | 5,512 |

The files are copied into a local `malaria_split/` folder so the generators can read them with `flow_from_directory`. The folder is recreated on every run to avoid mixing old and new files.

## Workflow

1. **Setup** — import the libraries, set the random seed (`SEED = 42`), and define the configuration (`IMG_SIZE = 128`, `BATCH_SIZE = 32`, `EPOCHS = 15`).
2. **Download the dataset** with `kagglehub`.
3. **Explore sample images** from both classes.
4. **Split** each class into train and test folders.
5. **Build the data generators**:
   - Training images use augmentation (rotation, shifts, shear, zoom, and flips).
   - Validation images are only rescaled.
   - Test images are rescaled with `shuffle=False`, so predictions stay aligned with the true labels.
6. **Build the CNN** — three convolution blocks with batch normalization and max pooling, followed by global average pooling and a small dense head.
7. **Compile and train** with Adam and early stopping / learning-rate reduction.
8. **Evaluate** on the held-out test set with a confusion matrix and a classification report.
9. **Predict a single image** with a reusable helper function.
10. **Optional Gradio app** — disabled by default.

All images are scaled to the `[0, 1]` range.

## Model

| Layer | Details |
| --- | --- |
| Conv block 1 | `Conv2D(32, 3×3, relu)` → BatchNorm → MaxPool |
| Conv block 2 | `Conv2D(64, 3×3, relu)` → BatchNorm → MaxPool |
| Conv block 3 | `Conv2D(128, 3×3, relu)` → BatchNorm → MaxPool |
| Head | GlobalAveragePooling → `Dense(128, relu)` → `Dropout(0.4)` → `Dense(1, sigmoid)` |

- **Total parameters:** 110,785 (110,337 trainable, 448 non-trainable)
- **Optimizer:** Adam with a learning rate of `0.0003`
- **Loss:** binary cross-entropy
- **Metric:** accuracy

### Training callbacks

- **EarlyStopping** monitors `val_loss` with `patience=5` and restores the best weights.
- **ReduceLROnPlateau** monitors `val_loss`, halves the learning rate after two stalled epochs, and stops at `min_lr=0.000001`.

## Results

Results from the saved GPU run:

| Metric | Value |
| --- | ---: |
| Test accuracy | **95.19%** |
| Test loss | **0.1340** |

### Classification report (test set)

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Parasitized | 0.97 | 0.93 | 0.95 | 2,756 |
| Uninfected | 0.94 | 0.97 | 0.95 | 2,756 |
| **Accuracy** | | | **0.95** | **5,512** |
| Macro avg | 0.95 | 0.95 | 0.95 | 5,512 |
| Weighted avg | 0.95 | 0.95 | 0.95 | 5,512 |

Precision and recall are balanced across the two classes. Parasitized recall (0.93) is slightly lower than Uninfected recall (0.97), so the model misses more parasitized cells than uninfected ones — the more costly error in a screening context.

The notebook also renders a confusion matrix and the full classification report.

> The saved outputs come from the original GPU run. Re-running the notebook may produce slightly different values because of random initialization and shuffling.

## Requirements

Python 3.11+ with:

```bash
pip install tensorflow kagglehub matplotlib seaborn scikit-learn
```

Optional, only for the Gradio demo:

```bash
pip install gradio
```

## How to Run

1. Open `Malaria_Cell_Classification.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. A GPU runtime is recommended — the saved run took roughly 2–2.5 minutes per epoch.
3. Run all cells from top to bottom.
4. Allow `kagglehub` to download the dataset if prompted.
5. Wait for training and evaluation to finish.

Training uses `EarlyStopping`, so it may stop before the 15-epoch limit.

### Optional Gradio demo

The Gradio app is **disabled by default** so the notebook runs end-to-end without opening a server. To use it:

1. Install Gradio.
2. Set `RUN_GRADIO = True` in the app cell.
3. Run the cell and upload a cell image.

Add `share=True` to `demo.launch()` only if you need a temporary public link.

## Project Files

| File | Description |
| --- | --- |
| `Malaria_Cell_Classification.ipynb` | Dataset download, image exploration, splitting, augmentation, CNN training, evaluation, and optional Gradio app |
| `README.md` | Project overview, results, and run instructions |
| `images/` | A few sample `Parasitized` and `Uninfected` cell images |

## Notes

- The split uses `SEED = 42` for reproducibility.
- Augmentation is applied only to the training data; validation and test data are only rescaled.
- The test set is never used for training or validation.
- For a stronger model, transfer learning (for example MobileNetV2 or EfficientNet) is a natural next step.

## Important Note

This project is for learning and demonstration only. It is not a medical device and must not replace diagnosis by a qualified medical professional.
