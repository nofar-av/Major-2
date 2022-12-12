import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#a
df = pd.read_csv(r'virus_data.csv')

#b
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.2, random_state = (73 + 98)) 

#Visualization and basic analysis - task
DataFrame = train[["PCR_01", "PCR_02", "spread"]] #TODO::NEEDS TO BE NORMALIZED

#Q1
g = sns.jointplot(DataFrame.PCR_01, DataFrame.PCR_02, hue=DataFrame.spread, palette='pastel')
plt.subplots_adjust(top=0.9)
plt.suptitle("PCR_01 and PCR_02 ")
plt.show()

#Q2
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
visualize_clf(knn, DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread, "kNN Model", "PCR_01", "PCR_02")


#model selection - Q3
from sklearn.model_selection import cross_validate

k_values  = list(range(1, 20, 2)) + list(range(20, 871, 85))
validation_scores= np.empty(shape=len(k_values), dtype=float)
train_scores = np.empty(shape=len(k_values), dtype=float)

for index, k in enumerate(k_values):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
  results = cross_validate(knn, DataFrame[["PCR_01", "PCR_02"]], y=DataFrame.spread, cv=8,  return_train_score=True)
  train_scores[index] = results["train_score"].mean()
  validation_scores[index] = results["test_score"].mean()

  

g = plt.semilogx(k_values, train_scores, color = 'orange', markersize = 15)
g = plt.semilogx(k_values, validation_scores, color = 'teal', markersize = 15)

plt.xlabel('k')
plt.ylabel('score')
plt.title('Cross Validation Train and Validation Scores (PCR_01 & PCR_02)')
plt.grid(True)
plt.legend(["train", "validation"], loc ="upper right")
plt.show()


#Q4
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
visualize_clf(knn, DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread, "kNN Model (k = 9)", "PCR_01", "PCR_02")

norm_test = prepare_data(train, test)
X_test = norm_test[["PCR_01", "PCR_02"]]
y_test = norm_test.spread
print("test score: " + str(knn.score(X_test, y_test)))

#Q5
from sklearn.neighbors import KNeighborsClassifier

for k in [1, 501]:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
  visualize_clf(knn, DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread, "kNN Model (k = " + str(k) + ")", "PCR_01", "PCR_02")

norm_test = prepare_data(train, test)
X_test = norm_test[["PCR_01", "PCR_02"]]
y_test = norm_test.spread
print("test score: " + str(knn.score(X_test, y_test)))

#model selection - Q6
from sklearn.model_selection import cross_validate

k_values  = list(range(1, 20, 2)) + list(range(20, 871, 85))
validation_scores= np.empty(shape=len(k_values), dtype=float)
train_scores = np.empty(shape=len(k_values), dtype=float)

for index, k in enumerate(k_values):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(norm_train, norm_train.spread)
  results = cross_validate(knn, norm_train, y=norm_train.spread, cv=8,  return_train_score=True)
  train_scores[index] = results["train_score"].mean()
  validation_scores[index] = results["test_score"].mean()

  
print(train_scores)
print(validation_scores)

g = plt.semilogx(k_values, train_scores, color = 'orange', markersize = 15)
g = plt.semilogx(k_values, validation_scores, color = 'teal', markersize = 15)

plt.xlabel('k')
plt.ylabel('score')
plt.title('Cross Validation Train and Validation Scores')
plt.grid(True)
plt.legend(["train", "validation"], loc ="lower left")
plt.show()


#Q7 decision trees  
#TODO::AN UGLY LOOKING TREE
from sklearn.tree import DecisionTreeClassifier, plot_tree

decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
decision_tree.fit(norm_train.drop(["risk","spread"], inplace= False, axis = 1), norm_train.risk)
trainAcc = np.sum(decision_tree.predict(norm_train.drop(["risk","spread"], inplace= False, axis = 1)) == norm_train.risk) * 100 / len(norm_train.risk)
# normalized_train = norm_test.drop(["risk","spread"], inplace= False, axis = 1)
print("train accuracy is "+ str(trainAcc))

plt.figure(figsize=(26,12))
plot_tree(decision_tree, feature_names=norm_train.columns, filled=True, fontsize=10)