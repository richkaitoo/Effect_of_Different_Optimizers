# Exploring the Impact of Feature Transformation on Machine Learning Model Performance

##  Abstract

This project investigates the impact of feature transformation techniques on machine learning model performance. We compare scaling, normalization, and encoding approaches to assess their effect on predictive accuracy and model robustness, using Root Mean Squared Error (RMSE) as the evaluation metric.

This study finds that feature engineering significantly affects model performance. Although the improvement wasn't dramatic, we observed a consistent increase in performance across models. This project demonstrates the importance of feature engineering in machine learning and highlights the need for careful consideration of transformation techniques.

---

##  Background
Feature transformation is a crucial step in the machine learning pipeline. It reshapes raw data into a format that enhances model learning. However, the impact of each transformation technique can vary depending on data structure, feature distribution, and the model applied.

This study aims to provide an empirical comparison of transformation effects across several algorithms, highlighting their advantages and limitations in predictive modeling.

---

##  Methodology
1. **Data Preparation** – Cleaned and preprocessed a dataset with missing and numerical features.  
2. **Transformation Techniques** – Applied scaling (StandardScaler) and encoding (One-Hot).  
3. **Model Training** – Trained baseline models including KNN, Linear Regression, and Decision Tree Regressor.  
4. **Evaluation Metrics** – Computed  RMSE for performance comparison.  
5. **Analysis** – Interpreted results to identify which transformations yield the best model performance.

---

##  Key Findings
- MinMax scaling significantly improved model performance.
- Normalization reduced variance sensitivity in regression models.
- Encoding techniques had varying effects depending on data sparsity.

---

## Technologies Used
- **Python**, **scikit-learn**, **pandas**, **NumPy**, **matplotlib**
- **Jupyter Notebook** for experimental documentation.

---

## Repository Structure

* `data/` contains the dataset, with subdirectories for raw and processed data.
* `src/` contains the source code, broken down into smaller notebooks or Python files for data loading, feature engineering, modeling, and utilities.
* `main.ipynb` is your main notebook that imports and uses the code from `src/`.
* `requirements.txt` lists the dependencies required to run your project.
* `README.md` provides an overview of your project, including instructions for running the code and reproducing the results.



