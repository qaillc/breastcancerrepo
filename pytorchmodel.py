import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Global scaler and label encoder for consistent preprocessing
scaler = StandardScaler()
label_encoder = LabelEncoder()
feature_columns = None  # To store feature columns from the training data

# Preload default files
DEFAULT_TRAIN_FILE = "patientdata.csv"
DEFAULT_PREDICT_FILE = "synthetic_breast_cancer_notreatmentcolumn.csv"
DEFAULT_LABEL_FILE = "synthetic_breast_cancer_data_withColumn.csv"

def main():
    global feature_columns

    st.title("Patient Treatment Prediction App")
    st.write("Upload patient data to train a model and predict treatments based on input data.")

    # Upload training data
    uploaded_file = st.file_uploader("Upload a CSV file for training", type="csv")
    if uploaded_file is None:
        st.write("Using default training data.")
        data = pd.read_csv(DEFAULT_TRAIN_FILE)
    else:
        data = pd.read_csv(uploaded_file)
    st.write("Training Dataset Preview:", data.head())

    # Check for Treatment column in training data
    if 'Treatment' not in data.columns:
        st.error("The training data must contain a 'Treatment' column.")
        return

    # Prepare Data
    X, y, input_dim, num_classes, feature_columns = preprocess_training_data(data)

    # Model Parameters
    hidden_dim = st.slider("Hidden Layer Dimension", 10, 100, 50)
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.01)  # Default set to 0.01
    epochs = st.number_input("Epochs", 1, 100, 20)

    # Model training
    if st.button("Train Model"):
        model, loss_curve = train_model(X, y, input_dim, hidden_dim, num_classes, learning_rate, epochs)
        plot_loss_curve(loss_curve)

    # Upload data for prediction
    st.write("Upload new data without the 'Treatment' column for prediction.")
    new_data_file = st.file_uploader("Upload new CSV file for prediction", type="csv")
    if new_data_file is None:
        st.write("Using default prediction data.")
        new_data = pd.read_csv(DEFAULT_PREDICT_FILE)
    else:
        new_data = pd.read_csv(new_data_file)
    st.write("Prediction Dataset Preview:", new_data.head())

    if 'model' in locals() and feature_columns is not None:
        # Align columns to match training data
        new_data_aligned = align_columns(new_data, feature_columns)
        
        if new_data_aligned is not None:
            predictions = predict_treatment(new_data_aligned, model)
            
            # Display Predictions in an Output Box
            st.subheader("Predicted Treatment Outcomes")
            prediction_output = "\n".join([f"Patient {i+1}: {pred}" for i, pred in enumerate(predictions)])
            st.text_area("Prediction Results", prediction_output, height=200)

            # Compare predictions with actual labels
            actual_data = pd.read_csv(DEFAULT_LABEL_FILE)
            if 'Treatment' in actual_data.columns:
                actual_labels = label_encoder.transform(actual_data['Treatment'])
                evaluate_model_performance(predictions, actual_labels)
            else:
                st.error("Actual labels file must contain a 'Treatment' column.")
        else:
            st.error("Unable to align prediction data to the training feature columns.")
    else:
        st.warning("Please train the model first before predicting on new data.")

def preprocess_training_data(data):
    global scaler, label_encoder

    # Label encode the 'Treatment' target column
    data['Treatment'] = label_encoder.fit_transform(data['Treatment'])
    y = data['Treatment'].values

    # Encode and standardize feature columns
    X = data.drop('Treatment', axis=1)
    feature_columns = X.columns  # Store feature columns for later alignment
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    # Standardize features
    X = scaler.fit_transform(X)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), X.shape[1], len(np.unique(y)), feature_columns

def align_columns(new_data, feature_columns):
    # Ensure the new data has the same columns as the training data
    missing_cols = set(feature_columns) - set(new_data.columns)
    extra_cols = set(new_data.columns) - set(feature_columns)
    
    # Remove any extra columns
    new_data = new_data.drop(columns=extra_cols)
    
    # Add missing columns with default value 0
    for col in missing_cols:
        new_data[col] = 0
    
    # Reorder columns to match the training data
    new_data = new_data[feature_columns]

    # Encode and standardize feature columns
    for col in new_data.select_dtypes(include=['object']).columns:
        new_data[col] = LabelEncoder().fit_transform(new_data[col])
    
    # Scale features
    new_data = scaler.transform(new_data)
    
    return torch.tensor(new_data, dtype=torch.float32)

def train_model(X, y, input_dim, hidden_dim, num_classes, learning_rate, epochs):
    # Model Definition
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Model, loss, optimizer
    model = SimpleNN(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    loss_curve = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())

    return model, loss_curve

def plot_loss_curve(loss_curve):
    plt.figure()
    plt.plot(loss_curve, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    st.pyplot(plt)

def predict_treatment(new_data, model, batch_size=32):
    model.eval()
    predictions = []

    # Run predictions in batches for large datasets
    with torch.no_grad():
        for i in range(0, new_data.size(0), batch_size):
            batch_data = new_data[i:i + batch_size]
            outputs = model(batch_data)
            _, batch_predictions = torch.max(outputs, 1)
            predictions.extend(batch_predictions.numpy())
    
    # Convert numeric predictions back to original label names
    return label_encoder.inverse_transform(predictions)

def evaluate_model_performance(predictions, actual_labels):
    # Ensure both predictions and actual_labels are consistently numeric
    if isinstance(predictions[0], str):
        actual_labels = label_encoder.inverse_transform(actual_labels)
    elif isinstance(predictions[0], int):
        actual_labels = label_encoder.transform(actual_labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions, average='weighted')
    recall = recall_score(actual_labels, predictions, average='weighted')
    f1 = f1_score(actual_labels, predictions, average='weighted')

    # Display metrics
    st.subheader("Model Evaluation Metrics")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(actual_labels, predictions)
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

if __name__ == "__main__":
    main()


