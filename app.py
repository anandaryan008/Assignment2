from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Extract all 16 input features from form
            features = [
                request.form.get('temperature_min'),
                request.form.get('temperature_max'),
                request.form.get('dissolved_oxygen_min'),
                request.form.get('dissolved_oxygen_max'),
                request.form.get('ph_min'),
                request.form.get('ph_max'),
                request.form.get('conductivity_min'),
                request.form.get('conductivity_max'),
                request.form.get('bod_min'),
                request.form.get('bod_max'),
                request.form.get('nitrate_min'),
                request.form.get('nitrate_max'),
                request.form.get('fecal_coliform_min'),
                request.form.get('fecal_coliform_max'),
                request.form.get('total_coliform_min'),
                request.form.get('total_coliform_max')
            ]
            
            # Validate inputs
            if not all(features):
                raise ValueError("All fields are required.")
            
            # Convert inputs to floats
            features = [float(value) for value in features]

            # Reshape input for prediction
            input_data = np.array(features).reshape(1, -1)
            prediction = model.predict(input_data)

            # Return prediction result
            return render_template("index.html", result=prediction[0])
        
        except ValueError as e:
            # Handle input errors
            return render_template("index.html", error=str(e))
    
    # Render form on GET request
    return render_template("index.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
