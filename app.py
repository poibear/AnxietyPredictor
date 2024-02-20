# TODO allow customization of dataset for ap_backend, add 404 w/ @app.errorhandler(404) page
import sqlite3
import os
from datetime import datetime
from ap_backend import AnxietyPredictor
from flask import Flask, redirect, render_template, request, url_for, flash

app = Flask(__name__, template_folder="templates")

ai = AnxietyPredictor()

model = ai.load_model()

if model is None: # no model file is present
    # build and train a model
    model = ai.build_model()
    ai.train_model(model)

@app.context_processor  # create a custom jinja variable for datetime
def datetime_variable():
    return {"datetime": datetime}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form")
def form():     
    topics_db = sqlite3.connect(os.path.join(os.getcwd(), "static/anxiety_factors_form.db"))
    topic_cursor = topics_db.cursor()
    
    topic_values = topic_cursor.execute(
        "SELECT main_topic, anxiety_factor, min_val, max_val, min_desc, topic_names.max_desc FROM topic_names"
    ).fetchall()
    
    topic_display = topic_cursor.execute(
        "SELECT main_topic, caption FROM topic_display"
    ).fetchall()
    
    topic_cursor.close()
    
    return render_template("survey.html", topic_values=topic_values, topic_display=topic_display)


@app.route("/results", methods=["GET", "POST"])
def results():
    if request.method == "GET":
        flash("You did not complete the survey. Try again.", "error")
        return redirect(url_for("form"))
        
    if request.method == "POST":        
        anxiety_params = {k: v for k, v in request.form.items()}        
        _, scaled_result = ai.predict_anxiety(model, anxiety_params)

        return render_template("results.html", anxiety_level=scaled_result)
    
    flash("You need to fill out the survey before getting results", "warning")
    return redirect(url_for("form"))


if __name__ == "__main__":
    host = "127.0.0.1"
    port = "8080"
    app.config["TEMPLATES_AUTO_RELOAD"] = True # reload on html change
    app.secret_key = 'supa secretz'
    app.debug = True
    
    app.run(host=host, port=port)