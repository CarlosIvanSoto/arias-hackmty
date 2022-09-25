from flask import Blueprint, render_template

app = Blueprint('routes-www', __name__)

@app.route('/gestos') # selector de avatar
def gestos():
    return render_template('gestos.html')

@app.route('/avatar') # selector de avatar
def avatar():
    return render_template('avatares.html')

@app.route('/register') # detector  
def register():
    return render_template('register.html')

@app.route('/login') # puntitos
def login():
    return render_template('login.html')

@app.route('/')
def home():
    return render_template('index.html')