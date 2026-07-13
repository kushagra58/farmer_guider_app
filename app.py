from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
RECOMMENDATIONS = {
    0: {
        "profile": "Low Rainfall, High Temperature (Arid/Semi-Arid Zone)",
        "irrigation": "Implement drip irrigation or rainwater harvesting. Avoid overhead sprinklers to prevent heavy evaporation losses.",
        "fertilizer": "Use organic compost and slow-release nitrogen fertilizers to retain soil moisture.",
        "soil_management": "Mulch the soil surface heavily to reduce evaporation and protect soil microbes from high heat."
    },
    1: {
        "profile": "High Rainfall, High Humidity (Tropical Zone)",
        "irrigation": "Focus on proper field drainage systems to prevent waterlogging and root rot. Extra watering rarely needed.",
        "fertilizer": "Apply fertilizers in split doses to avoid nutrient leaching caused by heavy rainfall.",
        "soil_management": "Grow cover crops during off-seasons to prevent heavy soil erosion from rain runoff."
    },
    2: {
        "profile": "High Potassium & Phosphorus, Moderate Weather (Nutrient-Rich/Alluvial Zone)",
        "irrigation": "Standard furrow or alternate-furrow irrigation works fine. Keep soil moisture consistent.",
        "fertilizer": "Scale back on P and K additions to prevent nutrient toxicity. Focus lightly on Nitrogen if needed.",
        "soil_management": "Practice crop rotation with legumes to balance high baseline soil nutrients naturally."
    },
    3: {
        "profile": "Low Nutrient, Balanced Climate (Loamy Plain Zone)",
        "irrigation": "Standard sprinkler irrigation or scheduled wetting intervals based on local crop selection.",
        "fertilizer": "Requires balanced N-P-K structural application. Utilize green manure or bio-fertilizers to boost organic matter.",
        "soil_management": "Incorporate deep tilling and organic biochar to structurally improve nutrient retention capacity."
    }
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_made = False
    result = None
    input_data = {}

    if request.method == 'POST':
        try:
            input_data = {
                'N': float(request.form['N']),
                'P': float(request.form['P']),
                'K': float(request.form['K']),
                'temperature': float(request.form['temperature']),
                'humidity': float(request.form['humidity']),
                'ph': float(request.form['ph']),
                'rainfall': float(request.form['rainfall'])
            }
            features = np.array([[
                input_data['N'], input_data['P'], input_data['K'],
                input_data['temperature'], input_data['humidity'],
                input_data['ph'], input_data['rainfall']
            ]])
            scaled_features = scaler.transform(features)
            cluster_id = int(model.predict(scaled_features)[0])
            result = RECOMMENDATIONS.get(cluster_id, {
                "profile": "Unknown Cluster",
                "irrigation": "Standard watering practices.",
                "fertilizer": "General N-P-K balanced application.",
                "soil_management": "Standard soil testing advised."
            })
            result['cluster_num'] = cluster_id
            prediction_made = True

        except Exception as e:
            result = {"error": f"Invalid input or server error: {str(e)}"}
            prediction_made = True

    return render_template('index.html', prediction_made=prediction_made, result=result, input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)