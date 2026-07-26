# Employee Attrition Prediction using Decision Tree and Random Forest Classification

## Objective
Build Decision Tree and Random Forest classification models to predict whether an employee is likely to leave the organization (attrition), based on demographic, professional, and work-related attributes, and compare their performance.

## Dataset Link
[IBM HR Analytics Employee Attrition & Performance Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hranalytics-attrition-dataset)

> The dataset is not included in this repository. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from the Kaggle link above and place it in the project root before running the notebook.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — data visualization
- `scikit-learn` — LabelEncoder, train/test split, Decision Tree Classifier, Random Forest Classifier, evaluation metrics

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, and identified numerical features (Age, MonthlyIncome, TotalWorkingYears, etc.), categorical features (BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, OverTime), and the target variable (`Attrition`).
2. **Data Preprocessing** — Checked for missing values (none found), removed unnecessary columns (`EmployeeCount`, `StandardHours`, `Over18` are constant for every row, and `EmployeeNumber` is just a unique ID), encoded the target and categorical columns using Label Encoding, and split the data 80/20 into training and testing sets (stratified on the target).
3. **Model Development** — Trained a Decision Tree Classifier and a Random Forest Classifier (100 estimators) on the same training data, then predicted attrition on the test set with both.
4. **Model Evaluation and Comparison** — Evaluated both models using Accuracy, Precision, Recall, and F1-score, generated a confusion matrix for each, and plotted feature importance for the Random Forest model.

## Results
| Metric | Decision Tree | Random Forest |
|---|---|---|
| Accuracy | ≈ 0.782 | ≈ 0.844 |
| Precision | ≈ 0.319 | ≈ 0.545 |
| Recall | ≈ 0.319 | ≈ 0.128 |
| F1-Score | ≈ 0.319 | ≈ 0.207 |

**Key observations:**
- Random Forest gets a noticeably higher accuracy than the Decision Tree (≈0.84 vs ≈0.78), which makes sense since it's combining 100 trees instead of relying on just one.
- But accuracy isn't the whole story, the Decision Tree actually gets a better recall and F1-score for the "Attrition = Yes" class than Random Forest does. So Random Forest is better at getting the overall prediction right, but it misses more of the actual attrition cases in this run.
- The top features from Random Forest's feature importance are `MonthlyIncome`, `Age`, `TotalWorkingYears`, `HourlyRate`, `DailyRate` and `MonthlyRate`.
- Both models struggle to catch every attrition case (low recall overall), probably because way more employees stay than leave in this dataset (imbalanced classes), and this seems to hurt Random Forest's recall a bit more than the Decision Tree's here.

## Model Comparison
Random Forest wins on accuracy and precision, but the Decision Tree actually wins on recall and F1-score for the minority ("Yes") class in this run. This is a good example of why accuracy alone can be misleading on an imbalanced dataset (way more "No" than "Yes" for attrition) — a model can rack up a high accuracy just by leaning towards predicting the majority class, without necessarily being great at catching the minority class we usually care more about (the people who are actually going to leave).

## Conclusion
In this project I built two classification models, a Decision Tree and a Random Forest (100 estimators), to predict whether an employee will leave the company using their demographic, professional and work-related info. After removing unnecessary columns (like EmployeeNumber and a few constant columns), encoding the categorical columns, and splitting the data 80/20, I trained both models on the same data and compared them using accuracy, precision, recall, F1-score, and confusion matrices.

Looking at just accuracy, Random Forest did better than the Decision Tree (≈0.84 vs ≈0.78). But when I looked closer at precision, recall and F1-score, the Decision Tree actually caught more of the real attrition cases (better recall/F1) even though its overall accuracy was lower, probably because the dataset is imbalanced, so a model can get high accuracy just by predicting "No" a lot, without necessarily being great at catching the "Yes" cases.

Random Forest usually outperforms a single Decision Tree because it trains lots of trees on random subsets of data and features, then averages/votes across them, cancelling out the mistakes any single tree makes and reducing overfitting. A Decision Tree's main limitation is that it tends to overfit the training data easily, memorizing patterns/noise that don't generalize well. Random Forest's main limitation is that its a lot less interpretable (you can't easily "see" the combined logic of 100 trees) and takes more time/memory to train. As seen here, Random Forest's higher accuracy doesn't automatically mean its better at every metric, recall on the minority class still needs to be checked separately, especially for problems like attrition where catching the "Yes" cases matters a lot.

## Bonus Challenge (Not Mandatory)
I tried changing `max_depth` for the Decision Tree (values: 3, 5, 7, and no limit) to see the effect. `max_depth=5` gave the best accuracy, while `max_depth=7` gave the best F1-score, both did better than letting the tree grow fully on at least one metric. This suggests a fully grown Decision Tree overfits a bit and doesn't generalize as well, while limiting the depth (somewhere around 5-7 here) helps keep it simpler and improves test performance, but going too small (like 3) starts to lose some of that benefit.
