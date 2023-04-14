import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

df = pd.read_csv("divorce.csv",delimiter=';')

print(df.info)
print(df.describe())


X = df.drop('Class', axis=1).values
y = df['Class'].values

print("Kiểm tra số lượng phần tử null \n" , df.isnull().sum())

print("Số lượng phần tử ", len(X))

print("Kiểm tra tập dử liệu có mất cân bằng ", df["Class"].value_counts())

acc_total_knn = 0
acc_total_dt = 0
acc_total_nb = 0


for i in range(10) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=(i+1)*10, shuffle=True)
    print("##########################Lần lặp", i+1, "###########################################\n\n")
    # Model K-Nearest Neighbors:
    print("############### Model K-Nearest Neighbors ############################")
    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_total_knn = acc_total_knn + acc_knn
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    precision_knn, recall_knn, f1_knn, _ = precision_recall_fscore_support(y_test, y_pred_knn)
    print("Accuracy K-NN: {:.2f}% \n".format(acc_knn * 100))
    print("Confusion Matrix K-NN:\n", cm_knn)
    print("Precision K-NN: {:.2f}% \n".format(precision_knn[0] * 100))
    print("Recall K-NN: {:.2f}% \n".format(recall_knn[0] * 100))
    print("F1 K-NN: {:.2f}% \n\n\n\n".format(f1_knn[0] * 100))

    # Model Decision Tree:
    print("############### Model Decision Tree ############################")
    model_dt = DecisionTreeClassifier(criterion = "entropy")
    model_dt.fit(X_train, y_train)
    y_pred_dt = model_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    acc_total_dt = acc_total_dt + acc_dt
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    precision_dt, recall_dt, f1_dt, _ = precision_recall_fscore_support(y_test, y_pred_dt)
    print("Accuracy Decision Tree: {:.2f}% \n".format(acc_dt * 100))
    print("Confusion Matrix Decision Tree:\n", cm_dt)
    print("Precision Decision Tree: {:.2f}% \n".format(precision_dt[0] * 100))
    print("Recall Decision Tree: {:.2f}% \n".format(recall_dt[0] * 100))
    print("F1 Decision Tree: {:.2f}% \n\n\n\n".format(f1_dt[0] * 100))

    # Model Naive Bayes:
    print("############### Model Naive Bayes ############################")
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    acc_total_nb = acc_total_nb + acc_nb
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    precision_nb, recall_nb, f1_nb, _ = precision_recall_fscore_support(y_test, y_pred_nb)
    print("Accuracy Naive Bayes: {:.2f}%\n".format(acc_nb * 100))
    print("Confusion Matrix Naive Bayes:\n", cm_nb)
    print("Precision Naive Bayes: {:.2f}%\n".format(precision_nb[0] * 100))
    print("Recall Naive Bayes: {:.2f}%\n".format(recall_nb[0] * 100))
    print("F1 Naive Bayes: {:.2f}%\n\n\n\n".format(f1_nb[0] * 100))
    print("\n\n\n\n\n")




print('Average accuracy of the model K-Nearest Neighbors : {:.2f}%'.format((acc_total_knn/10) * 100))
print('Average accuracy of the model Decision Tree : {:.2f}%'.format((acc_total_dt/10) * 100))
print('Average accuracy of the model Naive Bayes : {:.2f}%'.format((acc_total_nb/10) * 100))



print("Kiểm tra tập dử liệu có mất cân bằng ", df["Class"].value_counts())

