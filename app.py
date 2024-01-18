# app.py
from flask import Flask, render_template, request, jsonify
from ml_model import classify_mushroom

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()

    # Extracting features from the JSON
    capDiameter = int(data['capDiameter'])
    capShape = int(data['capShape'])
    capSurface = int(data['capSurface'])
    capColor = int(data['capColor'])
    doesBruiseOrBleed = int(data['doesBruiseOrBleed'])
    gillAttachment = int(data['gillAttachment'])
    gillColor = int(data['gillColor'])
    stemHeight = float(data['stemHeight'])
    stemWidth = float(data['stemWidth'])
    stemColor = int(data['stemColor'])
    hasRing = int(data['hasRing'])
    ringType = int(data['ringType'])
    habitat = int(data['habitat'])
    season = int(data['season'])

    # Calling machine learning model
    result = classify_mushroom(capDiameter, capShape, capSurface, capColor,
                                doesBruiseOrBleed, gillAttachment, gillColor,
                                stemHeight, stemWidth, stemColor,
                                hasRing, ringType, habitat, season)

    response_data = {'result': result}
    print("Sending response:", response_data)  # Add this line

    return jsonify(response_data), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
