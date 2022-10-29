from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')

@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})

def preprocessing(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dype=np.uint8)
    
if __name__ == "__main__":
    app.run()