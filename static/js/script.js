// =============================
// Intern Performance Predictor
// =============================

document.addEventListener("DOMContentLoaded", function () {

    console.log("Intern Performance Predictor Loaded");

    // -----------------------------
    // Form Validation
    // -----------------------------

    const form = document.querySelector("form");

    if (form) {

        form.addEventListener("submit", function (e) {

            const taskTime =
                parseFloat(document.querySelector("[name='task_time']").value);

            const feedback =
                parseFloat(document.querySelector("[name='feedback']").value);

            const attendance =
                parseFloat(document.querySelector("[name='attendance']").value);

            // Task Time Validation

            if (taskTime <= 0) {

                alert("Task Completion Time must be greater than 0.");

                e.preventDefault();

                return;

            }

            // Feedback Validation

            if (feedback < 1 || feedback > 5) {

                alert("Feedback Rating must be between 1 and 5.");

                e.preventDefault();

                return;

            }

            // Attendance Validation

            if (attendance < 0 || attendance > 100) {

                alert("Attendance must be between 0 and 100.");

                e.preventDefault();

                return;

            }

            // Disable Button

            const button = form.querySelector("button");

            button.disabled = true;

            button.innerHTML = "Predicting...";

        });

    }

});