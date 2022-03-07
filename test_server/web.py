from flask import Flask, render_template

app = Flask(__name__)

@app.route("/0")
def index_0():
    return render_template("0.html")

@app.route("/1")
def index_1():
    return render_template("1.html")

app.run()