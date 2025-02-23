# AI-Powered Applicant Tracking System (ATS)
## Overview
Applicant Tracking Systems (ATS) are essential for modern recruitment but often miss critical factors like cultural fit and genuine potential, leading to overlooked talent. Traditional ATS platforms struggle with resume overload and inefficient matching, and even with AI improvements, many systems fail to capture nuanced candidate attributes.

This project develops a lightweight ATS that uses TF-IDF for feature extraction and compares several classification models—including Logistic Regression and various Neural Network configurations—to improve matching accuracy and provide actionable feedback to job seekers.

## Data Description
### Dataset:
A comprehensive resume dataset from Kaggle containing structured fields such as career objectives, skills, education, work experience, certifications, and more.
- [Resume Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset/data)

### Data Processing:

- Resume Text: Combined from career_objective and skills columns.
- Job Matching Label: A binary label (job_match) created from the matched_score (1 if ≥ 0.7, else 0).

### Video
- [summative video](https://drive.google.com/file/d/1finMT_MKHzkGZvMY-PlIDJoLLMF9c_QF/view?usp=sharing)
  
## Methodology
1. Preprocessing:
- Use TF-IDF to convert combined text (resume & job information) into numerical features.
- Split data into training (60%), validation (20%), and test (20%) sets.
  
2. Models Implemented:
- Logistic Regression: A classical machine learning model tuned with C=10.
- Simple Neural Network (No Optimization): A baseline neural network with one hidden layer.
- Advanced Neural Networks:
   - Model 2: Neural network with Adam optimizer and early stopping.
   - Model 3: Neural network with RMSprop optimizer, L2 regularization (0.01), dropout (0.4), and enhanced early stopping.
   - Model 4: Neural network with SGD (with momentum), L2 regularization (0.005), dropout (0.4), a learning rate scheduler, and early stopping.

### Architecture:
The advanced neural networks use a deeper architecture with three hidden layers (128, 64, and 32 neurons) and an output layer of 1 neuron (with sigmoid activation). Biases are included by default in each Dense layer.

## Model Stats

| Training Instance               | Optimizer Used           | Regularizer Used | Epochs | Early Stopping | Number of Layers | Learning Rate | Accuracy | F1 Score | Precision | Recall |
|---------------------------------|--------------------------|------------------|--------|----------------|------------------|---------------|----------|----------|-----------|--------|
| **Logistic Regression**         | N/A (C=10)               | N/A              | N/A    | N/A            | N/A              | N/A           | 0.77     | 0.75     | 0.75      | 0.75   |
| **Simple NN (No Optimization)** | Adam (default)           | None             | 10     | No             | 1                | Default (~0.001) | 0.82  | 0.81     | 0.80      | 0.81   |
| **Model 2 (Adam Optimizer)**    | Adam                     | None             | 10     | Yes            | 3                | 0.01          | 0.81     | 0.80     | 0.80      | 0.80   |
| **Model 3 (RMSprop Optimization)** | RMSprop               | L2 (0.01)        | 50     | Yes            | 3                | 0.001         | 0.79     | 0.78     | 0.75      | 0.83   |
| **Model 4 (SGD with Momentum)** | SGD (with momentum)      | L2 (0.005)       | 50     | Yes            | 3                | 0.01          | 0.82     | 0.81     | 0.80      | 0.81   |


Notes:
The Simple NN is built without any explicit optimization techniques, serving as a baseline.
Model 2–4 introduce different optimization strategies and regularization methods to reduce overfitting and improve generalization.

## Discussion
- Best Overall Performance:
  The Simple NN and Model 4 show strong overall performance (Accuracy ≈ 0.82 and balanced F1, Precision, and Recall around 0.80), but the simple NN's training and validation loss graph showed drastic overfitting, making the SDG with momentum model superior.
- Trade-offs:
  Model 3, while having a slightly lower accuracy (0.79), demonstrates a higher recall (0.83), which might be preferred if the goal is to capture as many relevant resumes as possible.

### Instructions for Running the Notebook
1. Setup Environment:
- Install the required packages
```python
pip install numpy pandas scikit-learn tensorflow joblib matplotlib
```
2. Clone the Repository:
- Clone the project repository to your local machine.
  
3. Run the Notebook:
- Launch Jupyter Notebook:
```python
jupyter notebook
```
- Open Summative_Intro_to_ml_[Juliana_C_Holder]_assignment.ipynb and run the cells sequentially.
  
4. Model Saving & Loading:
- The best model (e.g., Model 4) is saved in the saved_models directory as best_optimized_model.keras.
- To load the model in a new session:
```python
from tensorflow.keras.models import load_model
model = load_model('saved_models/best_optimized_model.keras')
```
### Conclusion
This project demonstrates that even a lightweight, AI-powered ATS can significantly improve the accuracy and fairness of candidate matching compared to traditional systems. By leveraging TF-IDF and various classification models, we provide actionable insights for both recruiters and job seekers. Future enhancements could focus on integrating additional features (e.g., cultural fit) and further tuning model architectures to capture the subtleties of human potential.


