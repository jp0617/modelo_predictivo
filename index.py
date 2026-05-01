from flask import Flask, jsonify
import pandas as pd
from modelo.predict import init, predict, top5, team_stats

app = Flask(__name__)

init()


@app.route('/', methods=['GET'])
def home():
    result = top5()
    return jsonify({'top5': result})


@app.route('/team/<string:teamhome>/<string:teamaway>', methods=['GET'])
def disp(teamhome, teamaway):
    try:
        result = predict(teamhome, teamaway)
        return jsonify(result)
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
    app.run(debug=True)
