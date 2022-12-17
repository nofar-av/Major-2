import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import pylab
params = {'xtick.labelsize': 18,
 'ytick.labelsize': 18,
 'axes.titlesize' : 22,
 'axes.labelsize' : 20,
 'legend.fontsize': 18,
 'legend.title_fontsize': 22,
 'figure.titlesize': 24
 }
pylab.rcParams.update(params)

#a
df = pd.read_csv(r'virus_data.csv')

#b
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.2, random_state = (73 + 98)) 

#c - prepare data
from prepare import prepare_data

norm_train = prepare_data(train, train)
norm_test = prepare_data(train, test)

#Visualization and basic analysis - task
DataFrame = norm_train[["PCR_01", "PCR_02", "spread"]] #TODO::NEEDS TO BE NORMALIZED

#Q1
g = sns.jointplot(DataFrame.PCR_01, DataFrame.PCR_02, hue=DataFrame.spread, palette='pastel')
plt.subplots_adjust(top=0.9)
plt.suptitle("PCR_01 and PCR_02 ")
plt.show()

#Q2
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread)
visualize_clf(knn, DataFrame[["PCR_01", "PCR_02"]], DataFrame.spread, "kNN Model(k=1)", "PCR_01", "PCR_02")


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
  knn.fit(norm_train.drop(['risk', 'spread'], axis=1, inplace=False), norm_train.spread)
  results = cross_validate(knn, norm_train.drop(['risk', 'spread'], axis=1, inplace=False), y=norm_train.spread, cv=8,  return_train_score=True)
  train_scores[index] = results["train_score"].mean()
  validation_scores[index] = results["test_score"].mean()

  
print(train_scores)
print(validation_scores)
print("best k is:", k_values[np.argmax(validation_scores)])
print("its mean training and validation accuracies:", train_scores[np.argmax(validation_scores)], validation_scores[np.argmax(validation_scores)])

g = plt.semilogx(k_values, train_scores, color = 'orange', markersize = 15)
g = plt.semilogx(k_values, validation_scores, color = 'teal', markersize = 15)

plt.xlabel('k')
plt.ylabel('score')
plt.title('Cross Validation Train and Validation Scores')
plt.grid(True)
plt.legend(["train", "validation"], loc ="upper right")
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


#Q8
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth': range(1,20,2), 'min_samples_leaf': range(1,20,2)}
Decision_tree = DecisionTreeClassifier(criterion="entropy")
clf = GridSearchCV(Decision_tree, parameters, cv=8, return_train_score=True)
clf.fit(norm_train.drop(['risk', 'spread'], axis=1, inplace=False), norm_train.risk)
print(clf.best_estimator_)

pvt_train_mean = pd.pivot_table(pd.DataFrame(clf.cv_results_), values='mean_train_score', index='param_max_depth', columns='param_min_samples_leaf')
pvt_validation_mean = pd.pivot_table(pd.DataFrame(clf.cv_results_), values='mean_test_score', index='param_max_depth', columns='param_min_samples_leaf')

fig, axes = plt.subplots(1,2, figsize=(18, 5))
axes[0] = sns.heatmap(pvt_train_mean, vmin=0.5, vmax=1, cmap="hot", annot=True, ax=axes[0])
axes[0].set_title("mean train accuracy")
axes[0].set_xlabel("min_samples_leaf")
axes[0].set_ylabel("max_depth")

axes[1] = sns.heatmap(pvt_validation_mean, vmin=0.5, vmax=1, cmap="hot", annot=True, ax=axes[1])
axes[1].set_title("mean validation accuracy")
axes[1].set_xlabel("min_samples_leaf")
axes[1].set_ylabel("max_depth")
plt.show()

#Q9
decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=17)
decision_tree.fit(norm_train.drop(["risk","spread"], inplace= False, axis = 1), norm_train.risk)

print("test accuracy is:", decision_tree.score(norm_test.drop(["risk","spread"], inplace= False, axis = 1), norm_test.risk) * 100)

#Q10
X_train = np.array(norm_train[["PCR_01","PCR_02"]])
y_train = np.array(norm_train.spread)
compare_gradients(X_train, y_train, deltas=np.logspace(-9, -1, 12))

#Q11
for lr in np.logspace(-3, 0, 4):
  clf = SoftSVM(C=0.1, lr=lr)
  losses, accuracies = clf.fit_with_logs(X_train, y_train, max_iter=3000)
  plt.figure(figsize=(15, 8))
  plt.subplot(121), plt.grid(alpha=0.5), plt.title ("losses with lr=" + str(lr))
  plt.semilogy(losses), plt.xlabel("iteration"), plt.ylabel("loss")
  plt.subplot(122), plt.grid(alpha=0.5), plt.title ("accuracies with lr=" + str(lr))
  plt.plot(accuracies), plt.xlabel("iteration"), plt.ylabel("accuracy")

#Q12
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
C = 1e13
svm_2nd = Pipeline([('feature_mapping', PolynomialFeatures(2)), ('scaler', StandardScaler()), ('SVM', SoftSVM(C=C, lr=(1/(C*5e2))))])
svm_2nd.fit(X_train, y_train, SVM__max_iter=10000)
svm_3rd = Pipeline([('feature_mapping', PolynomialFeatures(3)), ('scaler', StandardScaler()), ('SVM', SoftSVM(C=C, lr=(1/(C*5e2))))])
svm_3rd.fit(X_train, y_train, SVM__max_iter=10000)

print("Train and test score 2nd deg: ", svm_2nd.score(X_train, y_train), svm_2nd.score(X_test, y_test))
print("Train and test score 2nd deg: ", svm_3rd.score(X_train, y_train), svm_3rd.score(X_test, y_test))
visualize_clf(svm_2nd, X_train, y_train, "Poly Model (2nd degree)", "PCR_01", "PCR_02")
visualize_clf(svm_3rd, X_train, y_train, "Poly Model (3rd degree)", "PCR_01", "PCR_02")

#Q13
train_acc = []
for i in range(5):
  svm_3rd = Pipeline([('feature_mapping', PolynomialFeatures(3)), ('scaler', StandardScaler()), ('SVM', SoftSVM(C=C, lr=(1/(C*5e2))))])
  svm_3rd.fit(X_train, y_train, SVM__max_iter=10000)
  train_acc.append(svm_3rd.score(X_train, y_train))
  visualize_clf(svm_3rd, X_train, y_train, "3rd Degree Poly Model (i=" + str(i) + ")", "PCR_01", "PCR_02")
  print("Train accuracy: ", train_acc[i])
print("mean and std: ", np.mean(train_acc), np.std(train_acc))

#Q15
from sklearn.svm import SVC
C = 1e4
for gamma in np.logspace(-5, 3, 9):
  clf = SVC(C=C, kernel="rbf", gamma=gamma)
  clf.fit(X_train, y_train)
  visualize_clf(clf, X_train, y_train, "SVC Model(RBF kernel) with gamma=" + str(gamma), "PCR_01", "PCR_02", marker_size=50)
  print("train and test accuracy ", clf.score(X_train, y_train), clf.score(X_test, y_test))

clf = SVC(C=C, kernel="rbf", gamma=1e4)
clf.fit(X_train, y_train)
visualize_clf(clf, X_train, y_train, "SVC Model(RBF kernel) with gamma=" + str(1e4), "PCR_01", "PCR_02", marker_size=10)
print("train and test accuracy ", clf.score(X_train, y_train), clf.score(X_test, y_test))

#Q17
from sklearn.model_selection import cross_validate

# gamma_values  = np.logspace(-1, 1, 14)
gamma_values  = np.linspace(1.5, 3, 16)
validation_scores= np.empty(shape=len(gamma_values), dtype=float)
train_scores = np.empty(shape=len(gamma_values), dtype=float)
C = 1e4
for index, gamma in enumerate(gamma_values):
  clf = SVC(C=C, kernel="rbf", gamma=gamma)
  clf.fit(X_train, y_train)
  results = cross_validate(clf, X_train, y=y_train, cv=8,  return_train_score=True)
  train_scores[index] = results["train_score"].mean()
  validation_scores[index] = results["test_score"].mean()

  

# g = plt.semilogx(gamma_values, train_scores, color = 'orange', markersize = 15)
# g = plt.semilogx(gamma_values, validation_scores, color = 'teal', markersize = 15)
g = plt.plot(gamma_values, train_scores, color = 'orange', markersize = 15)
g = plt.plot(gamma_values, validation_scores, color = 'teal', markersize = 15)


plt.xlabel('gamma')
plt.ylabel('score')
plt.title('SVC Model(RBF kernel) Cross Validation Train and Validation Scores(PCR_01 & PCR_02)')
plt.grid(True)
plt.legend(["train", "validation"], loc ="upper right")
plt.show()

print(gamma_values)
print("best gamma is: ", gamma_values[np.argmax(validation_scores)])
print("its validation score and train score: ", validation_scores[np.argmax(validation_scores)], train_scores[np.argmax(validation_scores)])

#Q18
best_gamma = 2.3
C=1e4
clf = SVC(C=C, kernel="rbf", gamma=best_gamma)
clf.fit(X_train, y_train)
print("Test accuracy: ",clf.score(X_test, y_test))
visualize_clf(clf, X_train, y_train, "SVC Model(RBF kernel) with gamma=" + str(best_gamma), "PCR_01", "PCR_02", marker_size=50)