from flask import Flask, render_template, session, redirect, request
import json
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'adfdlafaohgfoahgklaalksjflkjg'

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False, port = 9000, host="0.0.0.0")
