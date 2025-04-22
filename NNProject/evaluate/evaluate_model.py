import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Set paths
X_test_path = '/Users/nehasharma/Downloads/NNProject/Data/X.npy'
y_test_path = '/Users/nehasharma/Downloads/NNProject/Data/y.npy'
model_path = '/Users/nehasharma/Downloads/NNProject/models/lstm_classifier.keras'
output_dir = '/Users/nehasharma/Downloads/NNProject/evaluate'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)
print("Shape of X_test:", X_test.shape)

# Load model
model = load_model(model_path)

# Encode labels to one-hot
encoder = LabelEncoder()
y_test_encoded = encoder.fit_transform(y_test)
y_test_onehot = to_categorical(y_test_encoded)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=1)

# Predict classes
y_pred_probs = model.predict(X_test)
y_pred_encoded = np.argmax(y_pred_probs, axis=1)

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save evaluation results
evaluation_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.txt")
with open(evaluation_file, "w") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred_encoded)
labels = encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
plt.savefig(cm_path)
plt.close()

# Classification Report
report = classification_report(y_test_encoded, y_pred_encoded, target_names=labels)
report_path = os.path.join(output_dir, f"classification_report_{timestamp}.txt")
with open(report_path, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report)

# Summary
print(f"Evaluation results saved to: {evaluation_file}")
print(f"Confusion matrix image saved to: {cm_path}")
print(f"Classification report saved to: {report_path}")
