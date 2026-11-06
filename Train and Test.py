from EDA import df_encode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')


x= df_encode.drop('HeartDisease_1', axis = 1)
y = df_encode["HeartDisease_1"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {"Logistic Regression" : LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes" : GaussianNB(),
        "Decision Tree" : DecisionTreeClassifier(),
        "SVM" : SVC(kernel='rbf')
        }

result = []

for name,model in models.items():
    model.fit(X_trained_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    result.append({
        'model': name,
        'Accuracy': round(acc,4),
        'f1_score': round(f1,4)})
print(result)

import joblib
joblib.dump(models['Logistic Regression'],'LR_Heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(x.columns.tolist(),'columns.pkl')



