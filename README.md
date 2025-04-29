# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
## Feature Scaling:
```
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("C:/Users/admin/Documents/DS files/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/f7a0d55c-f2d1-4898-b049-b5a0a2278047)

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/bd5a5a99-c1d2-4186-b65c-2699dd3c145e)

```
df.columns
```
![image](https://github.com/user-attachments/assets/a0f89bc4-11e9-44ec-9153-a4f6ecd4c46f)

```
np.max(np.abs(df[['Height', 'Weight']]), axis=0)
```
![image](https://github.com/user-attachments/assets/1aef092e-bf7f-45de-98a6-9cee9d5c9f36)

### Standard Scaling:
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/95ec0056-16de-4637-9849-2967bf7aff95)

### Min-Max Scaling:
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/5ee76f9e-236e-4978-bfab-c585ded9c1c5)

### Maximum Absolute Scaling:
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/09ca1f00-e357-4729-874a-1c16a1cb67b3)

### Robust Scaling:
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/3633113d-758e-4491-837a-bbdc980b5a66)

## Feature Selection:
```
df=pd.read_csv("C:/Users/admin/Documents/DS files/income(1) (1).csv")
df
```
![image](https://github.com/user-attachments/assets/c3884230-14f9-49b5-8f0b-c2beb452b44d)

```
df.info()
```
![image](https://github.com/user-attachments/assets/b1849994-2762-4659-9b79-70986a55aa1c)

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/549a1811-49f3-4b7b-95cb-f6ac4f7a0a28)

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
categorical_columns
```
![image](https://github.com/user-attachments/assets/ff3d2a12-1587-4b1d-967e-39b9e49ade33)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/f4b6cb0f-3d05-444c-b4fa-811955de1889)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/360507c5-25ae-4ed9-a369-f40a595f2faa)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/453a6b78-bad4-45fb-8f8b-51bf478e8e86)

### Filter Method:
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/0fb92933-33a4-4569-828e-50e53433bfae)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4aba0aaf-e0ae-458b-9445-104d213b12fd)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]

print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/9ab775a5-731e-4296-b407-408f1d5dd97e)

### Model:
```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
       'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/3fa1e369-316d-4aa4-bfbc-12d1175d7930)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/c8df2d74-300f-4a18-b475-0617ebb99898)

```
!pip install skfeature-chappers
```

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')

df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ebe68206-79f3-4d93-b855-116d536805e6)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/961a6340-73a0-4f6a-8a55-25eb9896a8a7)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']

scores = fisher_score.fisher_score(X.to_numpy(), y.to_numpy())

ranked_features = np.argsort(scores)[::-1]

num_top_features = 5
top_features = X.columns[ranked_features[:num_top_features]]
print(f"Top features selected by Fisher Score: {list(top_features)}")

X_selected = X[top_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
```

### Anova:
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/7396b8c9-a41a-40cd-9f98-ee83c58b45d2)

### Wrapper Method:
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/1bb35dc0-5cc3-4561-92d5-ce553a41db2a)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/5ea9d5a7-e2a5-4413-8faa-4a29b7951a44)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/7d4ab35a-2305-40e4-b1bc-9ae1a522c046)

```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/5a1f158a-558d-4a37-aaa5-f1c4070abcd2)

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/cc434634-62a8-423d-841a-cba208b843be)


# RESULT:
Thus, the given data has been performed using Feature Scaling and Feature Selection process and saved to a file.
       
