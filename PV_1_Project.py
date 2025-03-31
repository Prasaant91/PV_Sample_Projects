import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from transformers import pipeline

# Step 1: Simulate Data (Replace this with actual data)
data_size = 1000
np.random.seed(42)
data = {
    "solar_irradiance": np.random.uniform(200, 1000, data_size),  # in W/m^2
    "temperature": np.random.uniform(15, 45, data_size),  # in Celsius
    "energy_output": lambda x: x["solar_irradiance"] * np.random.uniform(0.2, 0.25, data_size) - (x["temperature"] - 25) * 0.5,
}
data["energy_output"] = data["energy_output"](data)
# Simulate fault types
data["fault_type"] = np.random.choice(["No Fault", "Shading", "Soiling", "Inverter Issue", "Temperature Effects", "Electrical Disconnects"], data_size, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])
data = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Normalize the data
data["solar_irradiance_normalized"] = data["solar_irradiance"] / data["solar_irradiance"].max()
data["temperature_normalized"] = data["temperature"] / data["temperature"].max()

# Step 3: Split the Data for Performance Prediction
X = data[["solar_irradiance_normalized", "temperature_normalized"]]
y = data["energy_output"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Regression Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Regression Model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Regression Model RMSE: {rmse:.2f}")
print(f"Regression Model R^2 Score: {r2:.2f}")

# Step 6: Split the Data for Fault Detection
X_fault = data[["solar_irradiance_normalized", "temperature_normalized", "energy_output"]]
y_fault = data["fault_type"]
X_train_fault, X_test_fault, y_train_fault, y_test_fault = train_test_split(X_fault, y_fault, test_size=0.2, random_state=42)

# Step 7: Train the Fault Detection Model
fault_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
fault_model.fit(X_train_fault, y_train_fault)

# Step 8: Evaluate the Fault Detection Model
y_fault_pred = fault_model.predict(X_test_fault)
print("Fault Detection Classification Report:")
print(classification_report(y_test_fault, y_fault_pred, zero_division=0))

# Step 9: Enhanced Fault Fix Suggestions using DistilGPT

def suggest_fix_with_distilgpt(fault):
    generator = pipeline("text-generation", model="distilgpt2")
    prompt = (
        f"Solar panel systems rely on sunlight to generate electricity. "
        f"When a fault occurs, it can significantly reduce efficiency. The fault reported is: {fault}. "
        f"Explain the fault and provide actionable steps to resolve it, considering industry best practices."
    )
    suggestion = generator(prompt, max_length=200, temperature=0.7, top_k=50, num_return_sequences=1)[0]["generated_text"]
    # Post-process the response
    actionable_lines = [line.strip() for line in suggestion.split(".") if "fix" in line or "suggest" in line or "steps" in line]
    return " ".join(actionable_lines)

# Step 10: Build Dashboard
app = dash.Dash(__name__)

# Step 11: Dashboard Layout
app.layout = html.Div([
    html.H1("Solar Panel Performance and Fault Detection Dashboard", style={"textAlign": "center"}),

    dcc.Graph(id="scatter-plot", style={"width": "48%", "display": "inline-block"}),
    dcc.Graph(id="feature-importance", style={"width": "48%", "display": "inline-block"}),

    html.Div([
        html.Label("Select Data for Scatter Plot:"),
        dcc.Dropdown(
            id="scatter-feature",
            options=[
                {"label": "Solar Irradiance", "value": "solar_irradiance"},
                {"label": "Temperature", "value": "temperature"},
                {"label": "Energy Output", "value": "energy_output"},
            ],
            value="solar_irradiance",
        ),
    ], style={"width": "48%", "display": "inline-block"}),

    html.Div([
        html.Label("Fault Type and Suggested Fix"),
        dcc.Dropdown(
            id="fault-type-dropdown",
            options=[
                {"label": "No Fault", "value": "No Fault"},
                {"label": "Shading", "value": "Shading"},
                {"label": "Soiling", "value": "Soiling"},
                {"label": "Inverter Issue", "value": "Inverter Issue"},
                {"label": "Temperature Effects", "value": "Temperature Effects"},
                {"label": "Electrical Disconnects", "value": "Electrical Disconnects"},
            ],
            value="No Fault",
        ),
        html.Div(id="dynamic-fix-suggestion", style={"marginTop": 20})
    ], style={"width": "48%", "display": "inline-block"}),
])

# Step 12: Callbacks
@app.callback(
    Output("scatter-plot", "figure"),
    Input("scatter-feature", "value")
)
def update_scatter(selected_feature):
    fig = px.scatter(
        data, x=selected_feature, y="energy_output",
        title=f"Energy Output vs {selected_feature.capitalize()}",
        labels={selected_feature: selected_feature.capitalize(), "energy_output": "Energy Output"}
    )
    return fig

@app.callback(
    Output("feature-importance", "figure"),
    Input("scatter-feature", "value")
)
def update_feature_importance(_):
    feature_importances = model.feature_importances_
    fig = px.bar(
        x=X.columns, y=feature_importances,
        labels={"x": "Features", "y": "Importance"},
        title="Feature Importance"
    )
    return fig

@app.callback(
    Output("dynamic-fix-suggestion", "children"),
    Input("fault-type-dropdown", "value")
)
def dynamic_fix_suggestion(fault_type):
    suggestion = suggest_fix_with_distilgpt(fault_type)
    return f"Fault: {fault_type}. Suggested Fix: {suggestion}"

# Step 13: Run the App
if __name__ == "__main__":
    app.run_server(debug=True)
