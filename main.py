from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename

# Import your RAG functions
from rag_backend import load_and_index_documents, generate_rag_response

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

retriever_instance = load_and_index_documents()
chat_history = []

@app.route("/", methods=["GET", "POST"])
def index():
    global retriever_instance, chat_history

    # Handle file upload
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename.endswith(".pdf"):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            flash(f"Uploaded {filename} successfully!", "success")
            retriever_instance = load_and_index_documents()
            chat_history = []  # reset chat when new docs uploaded
        else:
            flash("Only PDF files allowed.", "danger")
        return redirect(url_for("index"))

    return render_template("index.html", chat_history=chat_history)

# JSON endpoint for chat
@app.route("/ask", methods=["POST"])
def ask():
    global retriever_instance, chat_history
    data = request.get_json()
    query = data.get("query", "")
    if retriever_instance:
        response = generate_rag_response(query, retriever_instance)
    else:
        response = "No documents indexed yet. Please upload a PDF first."
    chat_history.append({"user": query, "bot": response})
    return jsonify({"user": query, "bot": response})
    

if __name__ == "__main__":
    app.run(debug=True)
