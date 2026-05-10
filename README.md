🚢 Titanic Survival Prediction
Predict whether a passenger survived the Titanic disaster using Machine Learning.
This project is part of the Kaggle Titanic Competition.

📊 Dataset

Source: Kaggle Titanic Competition
Train rows: 891 passengers
Test rows: 418 passengers
Target: Survived (1 = Survived, 0 = Did Not Survive)


🔍 Features Used
ColumnDescriptionTypePclassPassenger class (1st, 2nd, 3rd)OrdinalSexGenderCategoricalAgeAge in yearsContinuousFareTicket fareContinuousEmbarkedPort of embarkation (S/C/Q)Categorical

❌ Features Dropped
ColumnReasonPassengerIdIdentifier column — no predictive valueNameIdentifier column — unique textTicketIdentifier column — random alphanumericCabin77% missing valuesSibSpWeak correlation with Survived (-0.04)ParchWeak correlation with Survived (+0.08)Embarked_QAlmost zero correlation with Survived (0.00)

⚙️ Process
1. Data Loading

Loaded train.csv and test.csv separately
Created working copy df_clean = df_train.copy()

2. Data Cleaning

Filled Age nulls with median
Filled Embarked nulls with mode
Filled Fare nulls (test.csv only) with median
Dropped Cabin — 77% missing values
Dropped identifier columns — PassengerId, Name, Ticket

3. EDA

Distribution plots for continuous columns (Age, Fare)
Count plots for categorical columns (Survived, Pclass, Sex, Embarked)
Correlation heatmap for feature selection

4. Encoding

Label Encoding → Sex (male=0, female=1)
One-Hot Encoding → Embarked (C, Q, S → dummy columns)

5. Feature Selection

Correlation analysis on numerical columns
Kept features above ±0.1 correlation threshold
Kept Age despite weak correlation — domain knowledge (women and children first)

6. Feature Scaling

Applied StandardScaler on continuous columns only
Scaled: Age, Fare
Not scaled: binary columns, ordinal columns, target variable
Used fit_transform on train, transform on test


🤖 Models & Results
ModelLocal AccuracyLogistic Regression79.9%Random Forest81.0% 🏆XGBoost81.0%

💡 Key Learnings

Identifier columns (PassengerId, Name, Ticket) must be dropped immediately
Domain knowledge matters — Age is important despite weak correlation
fit_transform on train, transform on test — never fit on test data
StandardScaler is better than MinMaxScaler when outliers exist
Fare has right skew — a few 1st class passengers paid very high fares


🛠️ Tech Stack

Python
Pandas, NumPy
Seaborn, Matplotlib
Scikit-learn
XGBoost
Joblib (model saving)


🚀 How To Run
bash# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib

# Run notebook
jupyter notebook titanic_survival_prediction.ipynb

📁 Files
├── train.csv                          ← training data
├── test.csv                           ← test data (no Survived column)
├── titanic_survival_prediction.ipynb  ← main notebook
├── titanic_lr_model.pkl               ← saved Logistic Regression model
├── titanic_rf_model.pkl               ← saved Random Forest model
├── titanic_xgb_model.pkl              ← saved XGBoost model
├── titanic_scaler.pkl                 ← saved StandardScaler
├── submission.csv                     ← Kaggle submission file
└── README.md

📈 Kaggle Submission

Competition: Titanic - Machine Learning from Disaster
Best model used: Random Forest
Submission file: submission.csv