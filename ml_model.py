# ml_model.py
import joblib
import numpy as np

def load_model():
    # Load your trained machine learning model using joblib
    model = joblib.load('C:\\Projects\\Mushroom_App\\model.pkl')
    return model

def classify_mushroom(capDiameter, capShape, capSurface, capColor, doesBruiseOrBleed, gillAttachment, gillColor, stemHeight, stemWidth, stemColor, hasRing, ringType, habitat, season):
    # Load the Random Forest model
    model = load_model()

    # Perform the classification
    features = np.array([[int(capDiameter), int(capShape), int(capSurface), int(capColor), int(doesBruiseOrBleed), int(gillAttachment), int(gillColor), int(stemHeight), int(stemWidth), int(stemColor), int(hasRing), int(ringType), int(habitat), int(season)]])
    result = model.predict(features)[0]

    # Convert the result to a human-readable label (e.g., "Safe" or "Poisonous")
    if(result == 0):
        return "Edible"
    else:
        return "Poisonous"
