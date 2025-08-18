import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

def main():
    df = pd.read_csv("data/features.csv")
    X = df.drop("label", axis=1)
    y = df["label"]
    feature_map = {name: f"f{i}" for i, name in enumerate(X.columns)}
    X = X.rename(columns=feature_map)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Original Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Original Model Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=12)
    onnx_path = "models/ecg_model.onnx"
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")

    onnx_quant_path = "models/ecg_model_quant.onnx"
    quantize_dynamic(onnx_path, onnx_quant_path, weight_type=QuantType.QInt8)
    print(f"Quantized ONNX model saved to {onnx_quant_path}")

if __name__ == "__main__":
    main()
