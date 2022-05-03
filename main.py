from flask import Flask, request, jsonify, json
import numpy

from models.lib.PPG_Peak_Identifier import SIGNAL_PEAKS_IDENTIFIER

app = Flask(__name__)

HEART_SIGNAL_PEAKS = SIGNAL_PEAKS_IDENTIFIER()

@app.route("/heartrate", methods=["POST"])
def result():
    result = json.loads(request.data)
    signal = result['signal']
    arr = numpy.array(signal)
    peaks = HEART_SIGNAL_PEAKS.identify_ppg_peaks(arr)
    response_data = {"peaks_count" : len(peaks.tolist())}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
