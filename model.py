from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load the data from divorce dataset
data = pd.read_csv("divorce.csv",sep=";")

# Split the data by hold-out 
X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.3, random_state=0)

# Train data using KNeighborsClassifier with k = 5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save the trained model to a file model.pkl
joblib.dump(model, 'model.pkl')
