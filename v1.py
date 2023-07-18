
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix



pd.set_option('display.max_columns', None)
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.head())

# train_df has 891 Instances with 11 features and the target variable of 'survived'
print(train_df.info())

# Some basic, descriptive statistics of the data set. We can see that approx 38% survived, the average age was
# approx 29.In addition, we see we have 714 instances of age, when we are supposed to have 891.
# That shows us we are missing data...
print(train_df.describe())

# Here I have calculated how many missing values are within each feature. We notice Age and Cabin have the most
# missing with 177 and 687 respectively.
missing = train_df.isnull().sum().sort_values(ascending=True)
print(missing)




# Data Preprocessing
# From my preliminary hypothesis, I believe the following features do not have a material impact on survivorship:
# PassengerID, Ticket, Name, Cabin. Hence, I will drop them from the dataset.
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(train_df.head())
print(train_df.info())




# Converting categorical data into numerical ones

# Converting Sex from male, female to male:0 and female:0
genders = {'male': 0, 'female': 1}
data = [train_df, test_df]
for i in data:
    i['Sex'] = i['Sex'].map(genders)






# Converting Embarked

# Dataset has 3 ports of embarkation, S, C, Q
# The counts below show 644 on Port S, 168 on Port C, and 77 on Port Q. As noted in the 'missing' table, we are missing
# 2 values in the embarkation feature

embark_count = train_df['Embarked'].value_counts()
print(embark_count)

ports = {'S': 0, 'C': 1, 'Q': 2}
data = [train_df, test_df]
for i in data:
    i['Embarked'] = i['Embarked'].map(ports)
print(train_df.head())

# Filling in 2 missing values for embarked. Filling in with the most common value of 'S': 0, count of 644 as seen above.
data = [train_df, test_df]
for i in data:
    i['Embarked'] = i['Embarked'].fillna(0.0)



# Filling in Age missing values
# Will fill in the missing values by Mean Imputation
mean_age = train_df['Age'].mean()
train_df['Age'].fillna(mean_age, inplace=True)
test_df['Age'].fillna(mean_age, inplace=True)


# Converting fare from float to int
data = [train_df, test_df]

for i in data:
    i['Fare'] = i['Fare'].fillna(0)
    i['Fare'] = i['Fare'].astype(int)



# Through this correlation matrix, we can see a potential area to analyze further: Survivorship x Age and
# Survivorship x Fare. Age appeared to have the highest correlation of 0.54 and Fare of 0.26.
# These two relationships provide us with a foundation to dive deeper.
corr_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
# plt.show()




# Machine Learning Models
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.copy()



# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)

tree.plot_tree(decision_tree)
plt.show()
# text_rep = tree.export_text(decision_tree)
# print(text_rep)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
log_pred = logreg.predict(X_test)


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)



accuracy_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
accuracy_regression = round(logreg.score(X_train, Y_train) * 100, 2)
accuracy_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(f'Decision Tree Score: ', accuracy_decision_tree)
print(f'Logistic Regression Score: ', accuracy_regression)
print(f'Random Forest Score: ', accuracy_random_forest)




# # Confusion Matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
predictions2 = cross_val_predict(logreg, X_train, Y_train, cv=3)
predictions3 = cross_val_predict(decision_tree, X_train, Y_train, cv=3)


confusion_matrix_dt = confusion_matrix(Y_train, predictions3)
confusion_matrix_lr = confusion_matrix(Y_train, predictions2)
confusion_matrix_rf = confusion_matrix(Y_train, predictions)

# Function to plot the confusion matrix with labels
def plot_confusion_matrix(confusion_matrix, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Confusion matrix with labels for Decision Tree
plot_confusion_matrix(confusion_matrix_dt, 'Decision Tree Confusion Matrix')

# Confusion matrix with labels for Logistic Regression
plot_confusion_matrix(confusion_matrix_lr, 'Logistic Regression Confusion Matrix')

# Confusion matrix with labels for Random Forest
plot_confusion_matrix(confusion_matrix_rf, 'Random Forest Confusion Matrix')