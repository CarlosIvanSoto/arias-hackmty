from flask import Blueprint, render_template

app = Blueprint('routes-www', __name__)

#add new task
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
def home():
    return render_template('index.html')