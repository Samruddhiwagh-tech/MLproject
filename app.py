from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = CustomData(
        gender=request.form.get("gender"),
        race_ethnicity=request.form.get("race_ethnicity"),
        parental_level_of_education=request.form.get("parental_level_of_education"),
        lunch=request.form.get("lunch"),
        test_preparation_course=request.form.get("test_preparation_course"),
        reading_score=int(request.form.get("reading_score")),
        writing_score=int(request.form.get("writing_score")),
    )

    final_df = data.get_data_as_data_frame()

    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(final_df)

    return render_template("index.html", results=result[0])


# 🔴 THIS IS WHY YOUR SERVER WAS NOT RUNNING
if __name__ == "__main__":
    app.run(debug=True)
