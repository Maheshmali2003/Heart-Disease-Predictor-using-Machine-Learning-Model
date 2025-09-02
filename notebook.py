import pandas as pd 

df = pd.read_csv("dataset.csv")
df.head()

'''
Heart Disease Dataset Features
age → Age of the patient (in years).
sex → Sex of the patient (1 = male, 0 = female).
cp (chest pain type) → Chest pain category (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic).
trestbps (resting blood pressure) → Resting blood pressure (in mm Hg) measured on admission.
chol (serum cholesterol) → Serum cholesterol level (mg/dl).
fbs (fasting blood sugar) → Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
restecg (resting electrocardiographic results) → Results of resting ECG (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy).
thalach (maximum heart rate achieved) → Maximum heart rate achieved during exercise.
exang (exercise induced angina) → Exercise-induced angina (1 = yes, 0 = no).
oldpeak → ST depression induced by exercise relative to rest (numeric value).
slope → Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).
ca (number of major vessels colored by fluoroscopy) → Count of major blood vessels (0–3).
thal → Thalassemia test result (1 = normal, 2 = fixed defect, 3 = reversible defect).
target → Presence of heart disease (1 = yes, 0 = no).
'''


df.shape


df['target'].value_counts()

#Train Test Split
from sklearn.model_selection import train_test_split

X = df.drop("target",axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

#Training & Evaluating Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report

models = {
    'LogisticRegression': LogisticRegression(),
    'Decision_Tree':DecisionTreeClassifier(),
    'Randome_Forest':RandomForestClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    # Train Model
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # Eval Model
    print(f"----------{name}----------")
    print("Report \n", classification_report(y_test, y_pred))
    print("Confusion \n", confusion_matrix(y_test, y_pred))
    print("\n\n")

#Selecting Best Model
model.fit(X_train,y_train)


#Save Model


import pickle
pickle.dump(model, open("model.pkl",'wb'))
