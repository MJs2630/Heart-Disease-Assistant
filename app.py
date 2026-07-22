from flask import Flask, request, render_template, redirect, url_for, session
import pickle

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for session storage

# Load model, scaler, and accuracy
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

# ---------------- HOME & AUTH ----------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login_page')
def login_page():
    return render_template('login_page.html')

@app.route('/signup_page')
def signup_page():
    return render_template('signup_page.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    # For demo: accept any username/password
    session['user'] = username
    session['prediction_history'] = []
    session['chat_history'] = []
    return redirect(url_for('dashboard'))

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    # For demo: just store username in session
    session['user'] = username
    session['prediction_history'] = []
    session['chat_history'] = []
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# ---------------- DASHBOARD ----------------

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
    last_prediction = session.get('last_prediction', None)
    history = session.get('prediction_history', [])
    return render_template('dashboard.html',
                           user=session['user'],
                           accuracy=accuracy,
                           last_prediction=last_prediction,
                           history=history)

# ---------------- PREDICTION ----------------

@app.route('/predict_form')
def predict_form():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('predict_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chol = float(request.form['cholestoral'])
        max_hr = float(request.form['Max_heart_rate'])
        exang = float(request.form['exercise_induced_angina'])
        oldpeak = float(request.form['oldpeak'])

        data = [age, sex, chol, max_hr, exang, oldpeak]
        final_input = scaler.transform([data])
        prediction = model.predict(final_input)[0]
        confidence = model.predict_proba(final_input)[0][prediction] * 100

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        prediction_record = {
            "Age": age,
            "Sex": "Male" if sex == 1 else "Female",
            "Cholestoral": chol,
            "Max Heart Rate": max_hr,
            "Exercise Angina": "Yes" if exang == 1 else "No",
            "Oldpeak": oldpeak,
            "Result": result,
            "Confidence": f"{confidence:.2f}%"
        }

        session['last_prediction'] = prediction_record
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        session['prediction_history'].append(prediction_record)

        return redirect(url_for('dashboard'))
    except Exception as e:
        return f"Error: {e}"

# ---------------- ASSISTANT ----------------

@app.route('/assistant')
def assistant():
    if 'user' not in session:
        return redirect(url_for('home'))
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('assistant.html', chat_history=session['chat_history'])

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']

    # Expanded demo answers
    if "cholesterol" in question.lower():
        answer = "Cholesterol is a fatty substance in your blood. High levels can increase heart disease risk."
    elif "prevent" in question.lower():
        answer = "You can reduce risk by exercising, eating healthy, avoiding smoking, and regular checkups."
    elif "symptoms" in question.lower():
        answer = "Common symptoms include chest pain, shortness of breath, fatigue, and irregular heartbeat."
    elif "diet" in question.lower():
        answer = "A heart-healthy diet includes fruits, vegetables, whole grains, lean proteins, and less salt."
    elif "exercise" in question.lower():
        answer = "Regular exercise strengthens your heart, lowers blood pressure, and improves circulation."
    elif "smoking" in question.lower():
        answer = "Smoking damages blood vessels and greatly increases heart disease risk. Quitting helps immediately."
    elif "stress" in question.lower():
        answer = "Chronic stress can raise blood pressure and harm your heart. Relaxation techniques are helpful."
    else:
        answer = "I’m still learning! Please ask about cholesterol, prevention, symptoms, diet, exercise, smoking, or stress."

    # Save Q&A in session chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({"question": question, "answer": answer})

    return render_template('assistant.html', chat_history=session['chat_history'])

# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)