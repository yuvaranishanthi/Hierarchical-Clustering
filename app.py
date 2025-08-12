from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from scipy.spatial.distance import cdist

app = Flask(__name__)

# Load saved model data
scaler = joblib.load("model/scaler.pkl")
features = joblib.load("model/features.pkl")
centroids = joblib.load("model/centroids.pkl")

# Cluster descriptions
cluster_descriptions = {
    0: ("High-income, high-spending", "Premium Customers who buy frequently"),
    1: ("Low-income, low-spending", "Minimal Shoppers"),
    2: ("High-income, low-spending", "Potential Upsell Targets"),
    3: ("Low-income, high-spending", "Value Seekers"),
    4: ("Mid-range income and spending", "Average Customers")
}

# Cluster colors
cluster_colors = {
    0: "#4CAF50",  # Green
    1: "#2196F3",  # Blue
    2: "#FFC107",  # Yellow
    3: "#FF5722",  # Orange
    4: "#9C27B0"   # Purple
}

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    color = None

    if request.method == 'POST':
        try:
            input_data = {f: float(request.form[f]) for f in features}

            # Scale input
            df_input = pd.DataFrame([input_data])
            scaled_input = scaler.transform(df_input)

            # Predict cluster by nearest centroid
            distances = cdist(scaled_input, centroids)
            cluster_label = np.argmin(distances)

            # Prepare result
            title, desc = cluster_descriptions.get(cluster_label, ("Unknown", "No description"))
            result = f"Cluster {cluster_label + 1} — {title} — {desc}"
            color = cluster_colors.get(cluster_label, "#333")

        except Exception as e:
            result = f"Error: {e}"
            color = "#000"

    return render_template("index.html", features=features, prediction=result, color=color)

if __name__ == '__main__':
    app.run(debug=True)

        