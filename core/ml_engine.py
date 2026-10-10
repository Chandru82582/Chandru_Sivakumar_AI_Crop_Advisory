"""Machine Learning Engine for Crop Yield Prediction and XAI.

Loads trained XGBoost models and SHAP explainer.
Provides predictions with confidence bounds and feature attribution.
"""
import joblib
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


class MLEngine:
    """Machine learning engine for crop recommendations and yield prediction."""

    def __init__(self, model_path: str = "artifact/crop_recommender.joblib"):
        """Initialize ML engine with trained models.

        Args:
            model_path: Path to saved model artifact
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact["model"]
        self.label_encoder = self.artifact["label_encoder"]
        self.features = self.artifact["features"]
        self.feature_stats = self.artifact["feature_stats"]
        self.known_states = set(self.artifact.get("known_states", []))

    def predict(self, features: dict) -> dict[str, Any]:
        """Predict top crops and yield metrics.

        Args:
            features: Dictionary with model features
                (N, P, K, pH, avg_temp_c, total_rainfall_mm, avg_humidity_percent, season_*)

        Returns:
            Dict with top 3 crop recommendations and confidence
        """
        try:
            X = self._prepare_features(features)
            probas = self.model.predict_proba(X)[0]

            top_k = 3
            top_indices = np.argsort(probas)[::-1][:top_k]

            crops = self.label_encoder.inverse_transform(
                self.model.classes_[top_indices]
            )
            confidences = probas[top_indices]

            predictions = []
            for crop, conf in zip(crops, confidences):
                predictions.append({
                    "crop": str(crop),
                    "probability": float(conf),
                    "confidence_pct": float(conf * 100)
                })

            overall_confidence = float(confidences[0])

            return {
                "predictions": predictions,
                "top_crop": predictions[0]["crop"] if predictions else None,
                "overall_confidence": overall_confidence,
                "is_high_confidence": overall_confidence > 0.4,
                "warnings": self._get_warnings(features)
            }

        except Exception as e:
            return {
                "error": str(e),
                "predictions": [],
                "warnings": [f"Prediction failed: {str(e)}"]
            }

    def _get_base_estimator(self):
        """Extract underlying tree estimator from CalibratedClassifierCV or pipeline."""
        if hasattr(self.model, 'calibrated_classifiers_') and self.model.calibrated_classifiers_:
            return self.model.calibrated_classifiers_[0].estimator
        if hasattr(self.model, 'estimator'):
            return self.model.estimator
        return self.model

    def get_shap_values(self, features: dict) -> dict[str, Any]:
        """Get SHAP feature importance for What-If analysis.

        Args:
            features: Dictionary with model features

        Returns:
            Dict with SHAP base value and feature contributions
        """
        try:
            X = self._prepare_features(features)
            base_est = self._get_base_estimator()

            if not hasattr(base_est, 'estimators_'):
                # Fallback to feature importance if tree estimators unavailable
                imp = self.get_feature_importance()
                contributions = [
                    {"feature": k, "shap_value": float(v), "impact": "positive"}
                    for k, v in sorted(imp.items(), key=lambda item: item[1], reverse=True)
                ]
                return {
                    "base_value": 0.0,
                    "contributions": contributions,
                    "top_5_features": [c["feature"] for c in contributions[:5]]
                }

            import shap

            tree = base_est.estimators_[0]
            explainer = shap.TreeExplainer(tree)
            shap_values = explainer.shap_values(X)

            probas = self.model.predict_proba(X)[0]
            top_class_idx = int(np.argmax(probas))

            if isinstance(shap_values, list):
                if top_class_idx < len(shap_values):
                    sv = shap_values[top_class_idx][0]
                else:
                    sv = shap_values[0][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[0, :, top_class_idx]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = np.zeros(len(self.features))

            exp_val = explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                base_value = float(exp_val[top_class_idx]) if top_class_idx < len(exp_val) else float(exp_val[0])
            else:
                base_value = float(exp_val)

            contributions = []
            for feature, value in zip(self.features, sv):
                val_float = float(value)
                contributions.append({
                    "feature": feature,
                    "shap_value": val_float,
                    "impact": "positive" if val_float > 0 else "negative"
                })

            contributions.sort(
                key=lambda x: abs(x["shap_value"]),
                reverse=True
            )

            return {
                "base_value": base_value,
                "contributions": contributions,
                "top_5_features": [c["feature"] for c in contributions[:5]]
            }

        except ImportError:
            return {
                "error": "SHAP library not installed",
                "base_value": 0,
                "contributions": []
            }
        except Exception as e:
            return {
                "error": str(e),
                "base_value": 0,
                "contributions": []
            }

    def _prepare_features(self, features: dict) -> pd.DataFrame:
        """Prepare and validate features for prediction.

        Args:
            features: Raw feature dictionary

        Returns:
            Validated feature dataframe
        """
        df = pd.DataFrame([features])

        required = ["N", "P", "K", "pH", "avg_temp_c", "total_rainfall_mm",
                   "avg_humidity_percent"]
        missing = [f for f in required if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        for col in self.features:
            if col not in df.columns:
                if col.startswith("season_"):
                    df[col] = 0
                else:
                    raise ValueError(f"Feature {col} not found in input")

        return df[self.features]

    def _get_warnings(self, features: dict) -> list[str]:
        """Get warnings about input data quality.

        Args:
            features: Feature dictionary

        Returns:
            List of warning strings
        """
        warnings = []

        if "pH" in features:
            pH = features["pH"]
            if not (0 <= pH <= 14):
                warnings.append(f"pH {pH} is outside valid range (0-14)")

        if "avg_humidity_percent" in features:
            humidity = features["avg_humidity_percent"]
            if not (0 <= humidity <= 100):
                warnings.append(f"Humidity {humidity}% is outside valid range (0-100%)")

        for col in ["N", "P", "K", "total_rainfall_mm"]:
            if col in features and features[col] < 0:
                warnings.append(f"{col} cannot be negative")

        return warnings

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importances from the model.

        Returns:
            Dict mapping feature names to importance values
        """
        try:
            base_est = self._get_base_estimator()
            if hasattr(base_est, 'feature_importances_'):
                importances = base_est.feature_importances_
                return {feat: float(imp) for feat, imp in zip(self.features, importances)}
            return {}
        except Exception:
            return {}

    def get_model_info(self) -> dict[str, Any]:
        """Get metadata about the trained model.

        Returns:
            Dict with model information
        """
        return {
            "model_name": self.artifact.get("model_name"),
            "created_at": self.artifact.get("created_at"),
            "n_features": len(self.features),
            "n_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "feature_names": self.features
        }
