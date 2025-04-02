# Air Quality Prediction Project

## Project Overview

This project aims to predict air quality using machine learning techniques. It involves data preprocessing, feature engineering, model training, and evaluation to build a predictive model for air quality assessment.

## Dataset

The project utilizes the "updated_pollution_dataset.csv" dataset, which contains various air quality parameters and corresponding air quality levels.

## Code Structure and Logic

1. **Data Loading and Preprocessing:**
   - Imports necessary libraries like pandas, numpy, scikit-learn, matplotlib, and seaborn.
   - Loads the dataset using `pd.read_csv`.
   - Checks for missing values and duplicates using `isnull().sum()` and `duplicated().sum()`.
   - Encodes categorical variables using `LabelEncoder`.
   - Separates features (X) and target variable (y).
   - Standardizes features using `StandardScaler`.
   - Splits data into training and testing sets using `train_test_split`.

2. **Model Training and Hyperparameter Tuning:**
   - Uses Support Vector Machine (SVM) as the prediction model (`SVC`).
   - Performs hyperparameter tuning using `GridSearchCV` to find the best model parameters.
   - Trains the model using the training data.

3. **Prediction and Evaluation:**
   - Makes predictions on the test data.
   - Evaluates the model using metrics like classification report, accuracy score, etc.

4. **Visualization:**
   - Creates visualizations to explore data patterns and model performance.
   - Uses `seaborn` and `matplotlib` to generate plots like count plots, correlation matrices, scatter plots, and box plots.

![image](https://github.com/user-attachments/assets/812d7733-694a-403d-ac87-085d5261237e)

  ![image](https://github.com/user-attachments/assets/b59ba5ad-242f-4e8c-8c41-3de0a5acadd0)
  
  ![image](https://github.com/user-attachments/assets/340bf08f-64e9-4fe7-a51a-a93d4cf6c214)


## Technology and Algorithms

- **Programming Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Algorithm:** Support Vector Machine (SVM) - A supervised learning algorithm used for classification and regression tasks. It works by finding an optimal hyperplane that separates data points into different classes.
- **Hyperparameter Tuning:** GridSearchCV - A technique used to find the best combination of hyperparameters for a model, improving its performance.
- **Evaluation Metrics:** Accuracy, Classification Report - Used to assess the model's prediction capabilities.


## Conclusion

This project demonstrates the application of machine learning for air quality prediction. The results and insights obtained can be valuable for environmental monitoring and decision-making.
