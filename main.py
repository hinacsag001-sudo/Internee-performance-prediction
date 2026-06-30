from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "random_forest_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Page
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        if model is None:
            return render_template(
                "result.html",
                score="Model Not Found",
                status="Please train the model first."
            )

        try:

            task_time = float(request.form["task_time"])
            feedback = float(request.form["feedback"])
            attendance = float(request.form["attendance"])

            prediction = model.predict([
                [
                    task_time,
                    feedback,
                    attendance
                ]
            ])

            score = round(float(prediction[0]), 2)

            # -----------------------------
            # Performance Category
            # -----------------------------

            if score >= 85:
                status = "Excellent"
                color = "green"

            elif score >= 70:
                status = "Average"
                color = "orange"

            else:
                status = "Needs Improvement"
                color = "red"

            return render_template(
                "result.html",
                score=score,
                status=status,
                color=color
            )

        except Exception as e:

            return render_template(
                "result.html",
                score="Error",
                status=str(e),
                color="red"
            )

    return render_template("predict.html")


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)