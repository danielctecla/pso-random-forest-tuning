import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """ Load the dataset """
    try:
        df = pd.read_csv('data/churn_bank_dataset.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found. Please ensure the file is in the 'data/' directory.")

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

# Load the dataset
X_train, X_test, y_train, y_test = load_data()


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
