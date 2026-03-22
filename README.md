### 1. Analysis and Classification of the “Digits” Dataset with PCA and SVM

This Python script performs a complete machine learning pipeline for exploratory analysis and classification of the well-known handwritten digit recognition dataset (“Digits”). The code integrates dimensional reduction techniques (PCA) with powerful classification algorithms (SVM).

### 2. Project Overview

The code is designed to load the digit data, explore the information variance, visualize relationships between classes in a two-dimensional space, and finally train a Support Vector Machine to classify images based on the extracted features.

### 3. Requirements and Dependencies

To run the script correctly and ensure proper execution, you must configure and install the Python packages listed in the “Dependencies and Requirements” section below.

### 4. Installation and Execution

To run the script on your local system, simply launch the terminal and follow these steps:

### - Prepare the environment (optional but recommended):
   It is always good practice to create and activate a virtual environment to keep your system dependencies clean.
   ```python
   python3 -m venv .venv
   
   # On Linux/macOS:
   source .venv/bin/activate  
   
   # On Windows:
   .venv\Scripts\activate
   ```

### - Running the Script and Results

<img width="800" height="500" alt="plot_pca" src="https://github.com/user-attachments/assets/b1349915-0027-4050-98d5-8d82c91603f1" />

<img width="800" height="600" alt="pca" src="https://github.com/user-attachments/assets/95f1e553-5c77-491b-93f5-fe3029cdc35d" />


### - SVM accuracy on PCA (pipeline): 0.5417

| Class / Metric | Precision | Recall | F1-score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 0.70 | 0.53 | 0.60 | 36 |
| **1** | 0.43 | 0.69 | 0.53 | 36 |
| **2** | 0.69 | 0.69 | 0.69 | 35 |
| **3** | 0.50 | 0.32 | 0.39 | 37 |
| **4** | 0.84 | 0.89 | 0.86 | 36 |
| **5** | 0.28 | 0.19 | 0.23 | 37 |
| **6** | 0.67 | 0.81 | 0.73 | 36 |
| **7** | 0.71 | 0.67 | 0.69 | 36 |
| **8** | 0.26 | 0.26 | 0.26 | 35 |
| **9** | 0.34 | 0.39 | 0.36 | 36 |
| | | | | |
| **accuracy** | | | 0.54 | 360 |
| **macro avg** | 0.54 | 0.54 | 0.53 | 360 |
| **weighted avg** | 0.54 | 0.54 | 0.53 | 360 |


