from src.explainability import explain_decision
from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)

# Ensure the 'reports' directory exists
REPORTS_DIR = os.path.join(os.getcwd(), "reports")
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Ensure a file is uploaded
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("upload.html", error="No file selected. Please upload a .txt file.")

        file = request.files["file"]

        # Ensure the file is a .txt file
        if file.filename == "" or not file.filename.endswith(".txt"):
            return "Please upload a valid .txt file", 400

        # Get the number of influential words from the form
        num_words = int(request.form.get("num_words", 10))

        # Save the file
        file_path = os.path.join(REPORTS_DIR, file.filename)
        file.save(file_path)

        # Load the file
        with open(file_path, 'r') as file:
            content = file.read()

        # Run prediction
        predictions = explain_decision(content, num_words, config_path='params.yaml')

        # Render result page with prediction and visualization
        return render_template(
            "result.html",
            housing_class=predictions["housing_pred"],
            eviction_class=predictions["eviction_pred"],
            housing_visualization_url=f"/reports/{predictions['housing_viz']}",
            eviction_visualization_url=f"/reports/{predictions['eviction_viz']}",
        )

    # Render upload form
    return render_template("upload.html")


@app.route("/reports/<path:filename>")
def serve_report(filename):
    """
    Serve files (e.g., visualizations) from the reports directory.
    """
    if os.path.exists(os.path.join(REPORTS_DIR, filename)):
        return send_from_directory(REPORTS_DIR, filename)
    else:
        return f"File {filename} not found.", 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))