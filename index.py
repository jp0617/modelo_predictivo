from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from modelo.predict import init, predict, top5, team_stats
from modelo.montecarlo import print_json_prediction
from modelo.hibrido.main import prediccion_personalizada

app = Flask(__name__)
CORS(app)

init()


@app.route('/', methods=['GET'])
def home():
    result = top5()
    return jsonify({'top5': result})


@app.route('/team/<string:teamhome>/<string:teamaway>', methods=['GET'])
def disp(teamhome, teamaway):
    try:
        result = print_json_prediction(teamhome, teamaway)
        result2 = predict(teamhome, teamaway)
        resultLogit = prediccion_personalizada(teamhome, teamaway)
        return {
            "montecarlo": result,
            "predict": result2,
            "logit&possion": resultLogit
        }
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/team/<string:team>', methods=['GET'])
def team_stats_endpoint(team):
    try:
        result = team_stats(team)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(debug=True,port=5001)
