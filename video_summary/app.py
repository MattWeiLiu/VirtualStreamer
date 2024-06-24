import json
import warnings
from flask import Flask, request, Response
from VideoSummarizer import VideoSummarizer
warnings.filterwarnings('ignore')

video_ai = VideoSummarizer()

# Creation of the Flask app
app = Flask(__name__)


# Flask route for Liveness checks
@app.route("/isalive")
def isalive():
    status_code = Response(status=200)
    return status_code


# Flask route for predictions
@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    info = request.get_json(silent=True, force=True)['instances'][0]
    if isinstance(info, dict):
        # info = {"Action": xxx, "Input": xxx, "open_id": xxx}
        if info["Action"] == "generate":
            summary = video_ai.generate(info["Input"])
            response_json = json.dumps({"predictions": [summary]})
        elif info["Action"] == "update_prompt":
            updated_prompt = video_ai.update_prompt(info["new_prompt"])
            response_json = json.dumps({"predictions": [updated_prompt]})
        elif info["Action"] == "get_prompt":
            response_json = json.dumps({"predictions": [video_ai.prompt]})
        return Response(response_json, status=200, mimetype="application/json")
    response_json = json.dumps({"predictions": "Error!!! The input must be in DICT format."})
    return Response(response_json, status=200, mimetype="application/json")


if __name__ == "__main__":
    # start api
    app.run(debug=False, host='0.0.0.0', port=8080)
