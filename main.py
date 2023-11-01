import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ChurnPredictionDataSpoofer:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
    
    def generate_data(self):
        # Generating age between 18 to 65
        age = np.random.randint(18, 66, self.n_samples)
        
        # Generating gender
        gender = np.random.choice(['Male', 'Female'], self.n_samples)
        
        # Generating tenure between 1 to 72 months
        tenure = np.random.randint(1, 73, self.n_samples)
        
        # Generating monthly charges between $20 to $100
        monthly_charges = np.random.uniform(20, 100, self.n_samples)
        
        # Generating feedback
        feedback = np.random.choice(['Good', 'Average', 'Poor'], self.n_samples)
        
        # Generating churn, with a random distribution
        churn = np.random.choice([0, 1], self.n_samples)
        
        # Combine all into a DataFrame
        data = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'Tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Feedback': feedback,
            'Churn': churn
        })
        
        return data

class ChurnDataPreprocessing:
    def __init__(self, data):
        self.data = data
    
    def encode_categorical(self):
        # Encoding 'Gender' and 'Feedback' using LabelEncoder
        le = LabelEncoder()
        self.data['Gender'] = le.fit_transform(self.data['Gender'])
        self.data['Feedback'] = le.fit_transform(self.data['Feedback'])
    
    def scale_numerical(self):
        # Scaling 'Age', 'Tenure', and 'MonthlyCharges' using StandardScaler
        scaler = StandardScaler()
        numerical_features = ['Age', 'Tenure', 'MonthlyCharges']
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])
    
    def split_data(self):
        # Splitting data into training and test sets
        X = self.data.drop('Churn', axis=1)
        y = self.data['Churn']
        return train_test_split(X, y, test_size=0.2, random_state=42)


class ChurnModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = RandomForestClassifier(random_state=42)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred)
        }
        return metrics


class ChurnDataVisualization:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def plot_feature_importance(self):
        # Extract feature importances from the model
        importances = self.model.feature_importances_
        
        # Create a DataFrame for the importances
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        
        # Sort the DataFrame by the importances
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()

# Generate data
data_gen = ChurnPredictionDataSpoofer(n_samples=1000)
data = data_gen.generate_data()

# Preprocess data
preprocessor = ChurnDataPreprocessing(data)
preprocessor.encode_categorical()
preprocessor.scale_numerical()
X_train, X_test, y_train, y_test = preprocessor.split_data()

# Build and train model
churn_model = ChurnModel(X_train, y_train)
churn_model.train_model()

# Evaluate model (Optional)
y_pred = churn_model.predict(X_test)
metrics = churn_model.evaluate(y_test, y_pred)
print(f"Model Metrics: {metrics}")

# Visualize feature importance
feature_names = X_train.columns.tolist()
viz = ChurnDataVisualization(churn_model.model, feature_names)
viz.plot_feature_importance()