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
