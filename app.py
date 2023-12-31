# TODO return DB values to survey to output in divs
import waitress
import sqlite3
import os
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__, template_folder="templates")

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


@app.route("/results")
def results():
    if request.method == "POST":
        
        return render_template("results.html")
    
    return render_template("results.html")


if __name__ == "__main__":
    host = "127.0.0.1"
    port = "8080"
    app.config["TEMPLATES_AUTO_RELOAD"] = True # reload on html change
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0 # no cache
    app.debug = True
    app.run(host=host, port=port)