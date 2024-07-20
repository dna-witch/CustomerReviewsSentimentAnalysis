import json
from flask import Flask, request, jsonify

from model import RNN_Model

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['input_review']
    pred_label, pred_prob = rnn.predict(input_text)

    result = {
        'prediction': pred_label,
        'probability': float(pred_prob)
    }

    print(f'Sentiment Prediction: {pred_label} with probability: {pred_prob:.3f}')

    return jsonify(result)

@app.route('/stats', methods=['GET'])
def stats():
    stats = rnn.get_model_stats()
    filepath = "results/model_stats.json"

    try: 
        with open(filepath, 'r') as f:
            json.dump(stats, f)
            results = f.read()
            return results
    except FileNotFoundError:
        return f'File not found: {filepath}'

if __name__ == '__main__':
    flaskPort = 8786
    rnn = RNN_Model()
    print('starting flask service...')
    app.run(host='0.0.0.0', port=flaskPort)
