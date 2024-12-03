from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/test')
def test():
    return jsonify(True)