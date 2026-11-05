from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import numpy as np

app = Flask(__name__)
# CORS allows your React/Vite frontend to communicate with this backend securely
CORS(app)

# Load the AI model we saved earlier
model = keras.models.load_model('box_office_predictor.keras')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # 1. Convert incoming dollar amounts to millions to match the AI's training data
        budget_millions = data['budget'] / 1000000.0
        marketing_millions = data['marketing'] / 1000000.0

        # 2. Format inputs as a dictionary matching the exact named Keras Input layers
        features = {
            'BUDGET_MILLIONS': np.array([budget_millions]),
            'MARKETING_MILLIONS': np.array([marketing_millions])
        }

        # 3. Get the prediction using the named dictionary
        prediction = model.predict(features, verbose=0)

        # 4. The AI predicts in "Millions". Multiply by 1,000,000 to get real dollars back.
        final_revenue = float(prediction[0][0]) * 1000000.0

        return jsonify({'status': 'success', 'predicted_revenue': final_revenue})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


if __name__ == '__main__':
    print("Starting AI Prediction Server on port 5000...")
    app.run(port=5000, debug=True)