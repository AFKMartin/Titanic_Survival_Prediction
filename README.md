# Titanic Survival Prediction (XGBoost)

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Kaggle](https://img.shields.io/badge/kaggle-competition-yellow)

This project predicts passenger survival on the **Titanic dataset** from [Kaggle](https://www.kaggle.com/c/titanic), using the **XGBoost** algorithm and hyperparameter tuning.

---

## Project Structure
```
Titanic_Survival_Prediction/
├── data/ # Raw and processed Titanic dataset
├── models/ # Trained models or saved checkpoints
├── notebooks/ # Jupyter notebooks for EDA and modeling
├── submissions/ # Kaggle submission CSVs
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Overview

The goal of this project is to build a machine learning model capable of predicting whether a passenger survived the Titanic disaster, based on features like age, gender, class, and fare.

Final mode performance: 92,17% accuracy.

The project includes:
- Data loading and exploration  
- Feature preprocessing and selection  
- Model training with **XGBoost**  
- **Hyperparameter optimization** using `GridSearchCV`  
- Model evaluation (accuracy, confusion matrix, and performance metrics)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AFKMartin/Titanic_Survival_Prediction.git
cd Titanic_Survival_Prediction
python -m venv venv
```
2. Create a Venv
```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

- Open the Jupyter notebooks in the `notebooks/` folder to explore the data, train the model, and make predictions.
- You can run each notebook step by step to reproduce preprocessing, model training, and evaluation.

## Technologies Used

| Library | Purpose |
|---------|---------|
| **NumPy**, **Pandas** | Data manipulation and analysis |
| **Matplotlib**, **Seaborn** | Data visualization |
| **Scikit-learn** | Model evaluation, train-test split, grid search |
| **XGBoost** | Gradient boosting classifier |

---

## Model Training

We tuned the XGBoost model using several hyperparameters:

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    eta=0.045454545454545456,
    max_depth=6,
    min_child_weight=1,
    gamma=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha = 0.05
)
```

## Model Performance

| Metric    | Score |
|-----------|-------|
| **Accuracy**  | 0.9217 |
| **Precision** | 0.9123 |
| **Recall**    | 0.8525 |
| **F1-Score**  | 0.8810 |

**Confusion Matrix:**  

|               | Predicted Not Survived | Predicted Survived |
|---------------|----------------------|------------------|
| **Actual Not Survived** | 113                  | 5                |
| **Actual Survived**     | 9                    | 52               |

**Prediction Counts:**  
- Survived: 57  
- Not Survived: 122

## Conclusion

- GBoost achieved strong predictive performance on the Titanic dataset (~92% accuracy).
- Feature engineering, like extracting titles and filling missing ages, significantly improved results.
- Future work: Explore ensemble methods, neural networks, or additional feature engineering to further improve accuracy (potentially approaching 95%).