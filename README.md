# CNN Malaria Classification

This project uses a Convolutional Neural Network (CNN) to classify microscopic cell images as either:

- **Parasitized**: The cell contains malaria parasites.
- **Uninfected**: The cell does not contain malaria parasites.

The project is written in a Jupyter Notebook and is ready to run in Google Colab.

## Project Files

- `Malaria_Cell_Classification.ipynb` — Downloads the dataset, prepares the images, trains the CNN, evaluates the model, and creates a Gradio interface.
- `images/` — Stores project images.

## Dataset

The notebook downloads the **Cell Images for Detecting Malaria** dataset from Kaggle using `kagglehub`. The dataset is divided into two folders:

- `Parasitized`
- `Uninfected`

The images are split into:

- 80% for training
- 20% for testing

Within the training data, 20% is also used as validation data.

## Model

The notebook builds a CNN using TensorFlow and Keras. The model includes:

- Convolution layers
- Batch normalization
- Max pooling layers
- Global average pooling
- Dense layers
- Dropout

The final layer uses sigmoid activation to classify each image as parasitized or uninfected.

## Results

In the saved notebook run, the model achieved:

- **Test accuracy:** 95.19%
- **Test loss:** 0.1340

The notebook also shows a confusion matrix and a complete classification report.

> These results may change when the notebook is run with another data split or environment.

## Requirements

The notebook uses:

- Python
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- KaggleHub
- Gradio

## How to Run

1. Open `Malaria_Cell_Classification.ipynb` in Google Colab.
2. Connect a GPU runtime for faster training.
3. Run the notebook cells from top to bottom.
4. Give KaggleHub permission to download the dataset if requested.
5. Wait for training to finish.
6. Run the final cell to start the Gradio interface.
7. Upload a cell image to test the model.

## How It Works

1. Download and load the malaria cell images.
2. Resize the images to 128 × 128 pixels.
3. Split the images into training, validation, and test sets.
4. Train the CNN to recognize image patterns.
5. Evaluate the model on images it has not seen before.
6. Upload a new image through the Gradio interface to receive a prediction.

## Important Note

This project is for learning and demonstration. It should not replace advice or diagnosis from a qualified medical professional.
