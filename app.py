from flask import Flask, jsonify, render_template, Response
from routes.www import app as www_bp
from routes.cv2 import app as cv2_bp
from routes.handDetect import app as hand_bp
import cv2
import mediapipe as mp


app = Flask(__name__)
#TEST CONECTION

# Aquí empiezan las rutas
#default check rest api route
@app.route('/ping')
def ping():
    return jsonify({"message": "Pong!"})
#add routes tasks
app.register_blueprint(www_bp)
app.register_blueprint(cv2_bp)

if __name__ == "__main__":
    app.run(debug=True)