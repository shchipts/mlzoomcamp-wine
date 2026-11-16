import joblib
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

artifact = joblib.load("models/random_forest.joblib")

model = artifact['model']
features = artifact['features']


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Random Forest prediction API is running"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({
                "error": "Expected a JSON object"
            }), 400

        missing = [f for f in features if f not in data]

        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        input_data = pd.DataFrame(
            [[data[f] for f in features]],
            columns=features
        )

        input_data = input_data.apply(
            pd.to_numeric,
            errors="raise"
        )

        if not np.isfinite(input_data.to_numpy()).all():
            return jsonify({
                "error": "Features must contain finite numeric values"
            }), 400

        prediction = model.predict(input_data)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
