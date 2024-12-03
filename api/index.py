from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/api')
def api():
    return jsonify([True])