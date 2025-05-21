# Fashion-MNIST

Project Description
This web application performs image clustering using machine learning techniques. It utilizes Principal Component Analysis (PCA) for dimensionality reduction, with using K-Means clustering algorithm to classify uploaded images. The application provides a user-friendly interface built with Streamlit that displays the clustering results in an elegant format.

Team Members
Hussah Almuzaini

Abdullah Aljubran

Majed Alsarawani

Jana Almalki

Features

🖼️ Image upload functionality (supports PNG, JPG, JPEG)

🎨 Custom styled interface with responsive design

🔍 Dual clustering with K-Means algorithm

📊 Clear visualization of clustering results

⚡ Fast processing with pre-trained models

Requirements
To run this application, you need:
   ```bach
streamlit
numpy
Pillow
scikit-learn
joblib

```

---

## 🛠️ Installation

**Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
```
Install the required packages:
```
pip install streamlit opencv-python numpy scikit-learn joblib pillow
```


## Usage
1- Run the application:
```
streamlit run app.py
```
2- The application will open in your default web browser at http://localhost:8501

3- Upload an image using the file uploader

4- View the results:

* Your uploaded image displayed in a styled frame

* K-Means cluster assignment

## How It Works
1- Image Processing:

* Converts image to grayscale

* Resizes to 28x28 pixels

* Flattens and normalizes pixel values

2- Dimensionality Reduction:

* Uses PCA to reduce features while preserving variance

3- Clustering:

* K-Means: Assigns to nearest cluster centroid

4- Results Display:

* Presents clustering result
