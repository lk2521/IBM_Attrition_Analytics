# 📋 IBM HR Analytics Employee Attrition & Performance

## 📈 Exploratory Data Analysis and Visualization

### Import Dependencies


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# Setting the 'searborn' style for graphs
plt.style.use("seaborn-v0_8")

# Read the dataset as Pandas DataFrame
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()

# Checking for missing values
df.isna().count()

df.info()

df.describe()

df.columns

df['Attrition']

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

df.head()

# Attrition: Yes -> 1 and No -> 0

### Remove the not useful columns

for column in df.columns:
    print(f'{column} : Unique Values = {df[column].nunique()}')
    print('=================================')

df.drop(['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'], axis=1, inplace=True)

len(df.columns)

def numerical_features_plots(df, feature_var, target_var = 'Attrition'):

    fig, ax = plt.subplots(ncols = 2, figsize=(16, 8))

    # boxplot for comparison
    sns.boxplot(x = target_var, y = feature_var, data=df, ax=ax[0], palette='muted')
    ax[0].set_title("Comparison of " + feature_var + " v/s " + target_var)

    # distribution plot
    ax[1].set_title("Distribution of "+feature_var)
    ax[1].set_ylabel("Frequency")
    sns.histplot(x = df[feature_var], ax=ax[1], palette='muted')

    plt.show()

def categorical_features_plots(df, feature_var, invert = False, target_var = "Attrition"):

    fig, ax = plt.subplots(ncols= 2, figsize = (14,6))

    # use invert variable for changing the axis whenever the names of categories overlap
    if invert == False:
        # countplot for distribution along with target variable
        sns.countplot(x = feature_var, data = df, hue = "Attrition",ax = ax[0], palette='muted')
    else:
        sns.countplot(y = feature_var, data = df,hue = "Attrition", ax = ax[0], palette='muted')

    ax[0].set_title("Comparison of " + feature_var + " v/s " + "Attrition")


    if invert == False:
        # barplot for plotting the effect of variable on attrition
        sns.barplot(x = feature_var, y = target_var , data = df, errorbar=('ci', 0), palette='muted')
    else:
        sns.barplot(x = target_var, y = feature_var, data = df, errorbar=('ci', 0), palette='muted')

    ax[1].set_title("Attrition rate by {}".format(feature_var))
    ax[1].set_ylabel("Average (Attrition)")
    plt.tight_layout()

    plt.show()

# For suppressing all warnings
import warnings
warnings.filterwarnings("ignore")

## Visualization of Variables

### Numerical Variables

### Age

# Visualizing the distribution of age of employees and their relation with Attrition
numerical_features_plots(df, 'Age')

### Daily Rate
numerical_features_plots(df, 'DailyRate')

### Monthly Income
numerical_features_plots(df, 'MonthlyIncome')

### Hourly Rate
numerical_features_plots(df, 'HourlyRate')

### Monthly Rate
numerical_features_plots(df, 'MonthlyRate')

### Percent Salary Hike
numerical_features_plots(df, 'PercentSalaryHike')
sns.countplot(df, y='PercentSalaryHike', hue='Attrition', palette='muted')
plt.show()

### Total Working Years
numerical_features_plots(df, 'TotalWorkingYears')

df['TotalWorkingYears'].nunique()

plt.figure(figsize=(15,5))
sns.countplot(df, x='TotalWorkingYears',hue='Attrition', palette='muted');

sns.lmplot(x = "TotalWorkingYears", y = "PercentSalaryHike", data=df, fit_reg=False, hue="Attrition", aspect=1.5)
plt.show()

### Distance From Home
numerical_features_plots(df, 'DistanceFromHome')

df['DistanceFromHome'].nunique()

plt.figure(figsize=(15,6))
sns.countplot(df, x='DistanceFromHome', hue='Attrition', palette='muted');

plt.figure(figsize=(15,6))
sns.barplot(x = 'DistanceFromHome', y = 'Attrition' , data = df, errorbar=('ci', 0), palette='muted');

### **Categorical Variables**

### Business Travel
categorical_features_plots(df, feature_var = 'BusinessTravel')

### Department
categorical_features_plots(df, feature_var='Department')

### Education Field and Education Level
df['EducationField'].value_counts()/len(df['EducationField'])*100

categorical_features_plots(df, feature_var='EducationField', invert=True)

plt.figure(figsize=(5,6))
sns.barplot(x = 'Attrition', y = 'EducationField' , data = df, hue='Education', errorbar=('ci', 0), palette='muted');

### Environment Satisfaction, Job Satisfaction, Relationship Satisfaction

df['EnvironmentSatisfaction'].value_counts()/len(df['EnvironmentSatisfaction'])*100

categorical_features_plots(df, feature_var='EnvironmentSatisfaction')

plt.figure(figsize=(7,5))
sns.barplot(y='Attrition', x='EnvironmentSatisfaction', hue='Department', data=df, errorbar=('ci', 0), palette='muted');

df['JobSatisfaction'].value_counts()/len(df['JobSatisfaction'])*100

categorical_features_plots(df, feature_var='JobSatisfaction')

plt.figure(figsize=(7,5))
sns.barplot(y='Attrition', x='JobSatisfaction', hue='Department', data=df, errorbar=('ci', 0), palette='muted');

df['RelationshipSatisfaction'].value_counts()/len(df['RelationshipSatisfaction'])*100

categorical_features_plots(df, feature_var='RelationshipSatisfaction')

### Job Involvement
df['JobInvolvement'].value_counts()/len(df['JobInvolvement'])*100

categorical_features_plots(df, feature_var='JobInvolvement')

plt.figure(figsize=(10,5))
sns.barplot(y='Attrition', x='JobInvolvement', hue='JobSatisfaction', data=df, errorbar=('ci', 0), palette='muted');

### Job Level
df['JobLevel'].value_counts()/len(df['JobLevel'])*100

categorical_features_plots(df, feature_var='JobLevel')

### Job Role
df['JobRole'].value_counts()/len(df['JobRole'])*100

categorical_features_plots(df, feature_var='JobRole', invert=True)

### Marital Status
categorical_features_plots(df, feature_var='MaritalStatus')

### Number of Companies Worked
categorical_features_plots(df, feature_var='NumCompaniesWorked')

### OverTime
df['OverTime'].value_counts()/len(df['OverTime'])*100

categorical_features_plots(df, feature_var='OverTime')

### Performance Rating
df['PerformanceRating'].value_counts()/len(df['PerformanceRating'])*100

categorical_features_plots(df, feature_var='PerformanceRating')

### Stock Option Level
df['StockOptionLevel'].value_counts()/len(df['StockOptionLevel'])*100

categorical_features_plots(df, feature_var='StockOptionLevel')

### Work Life Balance

df['WorkLifeBalance'].value_counts()/len(df['WorkLifeBalance'])*100

categorical_features_plots(df, feature_var='WorkLifeBalance')

### Training Times Last Year

categorical_features_plots(df, feature_var='TrainingTimesLastYear')

### Gender
categorical_features_plots(df, feature_var='Gender')

sns.boxplot(x = 'Gender', y = 'MonthlyIncome', data=df, palette='muted')
plt.title('MonthlyIncome vs Gender Box Plot', fontsize=20)
plt.ylabel('MonthlyIncome', fontsize=16)
plt.xlabel('Gender', fontsize=16)
plt.show()

fig,ax = plt.subplots(2,3, figsize=(20,20))               # 'ax' has references to all the four axes
plt.suptitle("Comparision of various factors vs Gender", fontsize=30)
sns.barplot(x = 'Gender', y = 'DistanceFromHome',hue = 'Attrition', data = df, ax = ax[0,0], errorbar=('ci', 0), palette='muted')
sns.barplot(x = 'Gender', y = 'YearsAtCompany',hue = 'Attrition', data = df, ax = ax[0,1], errorbar=('ci', 0), palette='muted')
sns.barplot(x = 'Gender', y = 'TotalWorkingYears',hue = 'Attrition', data = df, ax = ax[0,2], errorbar=('ci', 0), palette='muted')
sns.barplot(x = 'Gender', y = 'YearsInCurrentRole',hue = 'Attrition', data = df, ax = ax[1,0], errorbar=('ci', 0), palette='muted')
sns.barplot(x = 'Gender', y = 'YearsSinceLastPromotion',hue = 'Attrition', data = df, ax = ax[1,1], errorbar=('ci', 0), palette='muted')
sns.barplot(x = 'Gender', y = 'NumCompaniesWorked',hue = 'Attrition', data = df, ax = ax[1,2], errorbar=('ci', 0), palette='muted')
plt.show();


df.drop(['HourlyRate', 'MonthlyRate', 'MaritalStatus', 'PerformanceRating'], axis=1, inplace=True)

df.head()

df_corr = df.copy()

categorical_cols = [column for column in df_corr.columns if df_corr[column].dtype==object]

categorical_cols

for col in categorical_cols:
    df_corr[col] = label_encoder.fit_transform(df_corr[col])

df_corr.head()

plt.figure(figsize=(24,16))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)

df_cat = pd.get_dummies(df_corr[categorical_cols], drop_first=False, dtype='uint8')

df_num = df_corr.drop(categorical_cols, axis=1)

df_num

df_cat.head()

df_new = pd.concat([df_num, df_cat], axis=1)

df_new.head()

df_new.drop('Attrition', axis=1).corrwith(df_new.Attrition).sort_values().plot(kind='barh', figsize=(10, 10))
plt.xlabel('Correlation');


"""## Machine Learning Models"""

df_new['Attrition'].value_counts()

"""* Attrition categories are so imbalanced, so we need to do stratified sampling."""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Seperate features and targets
X = df_new.drop('Attrition', axis=1)
y = df_new.Attrition

# Split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Looking into splitted data
y_train.value_counts()/len(y_train)

y_test.value_counts()/len(y_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    group_names = ['True Negative','False Positive','False Negative','True Positive']

    print("TRAINING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print("CONFUSION MATRIX:\n")
    train_cf_matrix = confusion_matrix(y_train, y_train_pred)
    group_counts = ["{0:0.0f}".format(value) for value in train_cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(5,3))
    sns.heatmap(train_cf_matrix, cmap='Blues', annot=labels, fmt='')
    plt.show()
    print(f"\nACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}\n")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("\n\nTESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print("CONFUSION MATRIX:\n")
    test_cf_matrix = confusion_matrix(y_test, y_test_pred)
    group_counts = ["{0:0.0f}".format(value) for value in test_cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize=(5,3))
    sns.heatmap(test_cf_matrix, cmap='Blues', annot=labels, fmt='')
    plt.show()
    print(f"\nACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}\n")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

"""## Model 1: Logistic Regression"""

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear', penalty='l1')
lr_clf.fit(X_train_scaled, y_train)

evaluate(lr_clf, X_train_scaled, X_test_scaled, y_train, y_test)

from sklearn.metrics import precision_recall_curve, roc_curve

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Precision/Recall Tradeoff")


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


precisions, recalls, thresholds = precision_recall_curve(y_test, lr_clf.predict(X_test_scaled))
plt.figure(figsize=(14, 25))
plt.subplot(4, 2, 1)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.subplot(4, 2, 2)
plt.plot(precisions, recalls)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve: precisions/recalls tradeoff")
plt.savefig('pr_tradeoff.png');

plt.subplot(4, 2, 3)
fpr, tpr, thresholds = roc_curve(y_test, lr_clf.predict(X_test_scaled))
plot_roc_curve(fpr, tpr)

scores_dict = {
    'Logistic Regression': {
        'Train': roc_auc_score(y_train, lr_clf.predict(X_train_scaled)),
        'Test': roc_auc_score(y_test, lr_clf.predict(X_test_scaled)),
    },
}

"""## Model 2: Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, bootstrap=False, #class_weight={0:stay, 1:leave}
                               )
rf_clf.fit(X_train, y_train)
evaluate(rf_clf, X_train, X_test, y_train, y_test)

# Create a dictionary of hyperparameters to find best combinations among them
param_dict = dict(
    n_estimators= [100, 500, 900],
    max_features= ['auto', 'sqrt'],
    max_depth= [2, 3, 5, 10, 15, None],
    min_samples_split= [2, 5, 10],
    min_samples_leaf= [1, 2, 4],
    bootstrap= [True, False]
)

# Search for best hyperparameters
rf_clf = RandomForestClassifier(random_state=42)
search = GridSearchCV(rf_clf, param_grid=param_dict, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
search.fit(X_train, y_train)

rf_clf = RandomForestClassifier(**search.best_params_, random_state=42)
rf_clf.fit(X_train, y_train)
evaluate(rf_clf, X_train, X_test, y_train, y_test)

precisions, recalls, thresholds = precision_recall_curve(y_test, rf_clf.predict(X_test))
plt.figure(figsize=(14, 25))
plt.subplot(4, 2, 1)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.subplot(4, 2, 2)
plt.plot(precisions, recalls)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve: precisions/recalls tradeoff");

plt.subplot(4, 2, 3)
fpr, tpr, thresholds = roc_curve(y_test, rf_clf.predict(X_test))
plot_roc_curve(fpr, tpr)

scores_dict['Random Forest'] = {
        'Train': roc_auc_score(y_train, rf_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, rf_clf.predict(X_test)),
    }

"""## Support Vector Machine"""

from sklearn.svm import SVC

svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train_scaled, y_train)

evaluate(svm_clf, X_train_scaled, X_test_scaled, y_train, y_test)

svm_clf = SVC(random_state=42)

param_dict = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
]

search = GridSearchCV(svm_clf, param_grid=param_dict, scoring='roc_auc', cv=3, refit=True, verbose=1)
search.fit(X_train_scaled, y_train)

svm_clf = SVC(**search.best_params_)
svm_clf.fit(X_train_scaled, y_train)

evaluate(svm_clf, X_train_scaled, X_test_scaled, y_train, y_test)

precisions, recalls, thresholds = precision_recall_curve(y_test, svm_clf.predict(X_test_scaled))
plt.figure(figsize=(14, 25))
plt.subplot(4, 2, 1)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.subplot(4, 2, 2)
plt.plot(precisions, recalls)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.title("PR Curve: precisions/recalls tradeoff");


plt.subplot(4, 2, 3)
fpr, tpr, thresholds = roc_curve(y_test, svm_clf.predict(X_test_scaled))
plot_roc_curve(fpr, tpr)

scores_dict['Support Vector Machine'] = {
        'Train': roc_auc_score(y_train, svm_clf.predict(X_train_scaled)),
        'Test': roc_auc_score(y_test, svm_clf.predict(X_test_scaled)),
    }

"""## Comparison of Models Performances"""

for key in scores_dict.keys():
    print(f"{key.upper()}:\n")
    print(f"Train ROC_AUC_SCORE: {scores_dict[key]['Train']}")
    print(f"Test ROC_AUC_SCORE: {scores_dict[key]['Test']}")
    print("\n=================================================\n")

scores_df = pd.DataFrame(scores_dict)
scores_df.plot(kind='barh', figsize=(15, 8), cmap='viridis')
plt.xlabel("ROC_AUC_SCORE");