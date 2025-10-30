from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ Allow frontend (HTML/React) to call the API

# Load your trained model
model = joblib.load("placementmodel.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract features in correct order (must match model training)
    features = np.array([[
        data["CGPA"],
        data["Internships"],
        data["Projects"],
        data["Workshops_Certifications"],
        data["AptitudeTestScore"],
        data["SoftSkillsRating"],
        data["ExtracurricularActivities"],
        data["PlacementTraining"],
        data["SSC_Marks"],
        data["HSC_Marks"]
    ]])

    # Predict class and probability
    prediction = model.predict(features)[0]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])
    else:
        probability = None  # In case the model doesn't support probability

    return jsonify({
        "predicted_class": int(prediction),
        "placement_probability": probability
    })



if __name__ == "__main__":
    app.run(debug=True)
