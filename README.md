**Solar Panel Performance and Fault Detection Dashboard
**This project presents a comprehensive solution for modeling, analyzing, and visualizing the performance of solar photovoltaic (PV) systems. Leveraging simulated data, machine learning, and natural language processing, it showcases how energy analytics and intelligent fault detection can be seamlessly integrated into a single platform. The dashboard provides interactive tools to understand how irradiance and temperature affect energy output, detect various common PV faults, and even generate AI-powered maintenance suggestions in natural language.

The entire application is built using Python, combining the capabilities of libraries like Pandas, NumPy, scikit-learn, and Transformers (Hugging Face) with the interactivity and visual richness of Dash and Plotly. It serves as a starting point for energy analysts, data scientists, or engineers interested in building smart solar monitoring systems or predictive maintenance tools using open-source technologies.

**Objective
**The aim of this project is to simulate a real-world solar PV monitoring environment where energy production can be predicted based on weather conditions, and various system faults can be automatically detected and resolved using intelligent suggestions. In large-scale solar farms, issues like shading, soiling, inverter problems, and thermal inefficiencies can lead to energy loss. Identifying these issues early—and suggesting timely interventions—can help operators save energy, reduce downtime, and extend equipment life.

Rather than building this functionality separately, this project integrates all three layers—data simulation, ML modeling, and fault interpretation—into one coherent dashboard. This makes it an ideal proof-of-concept (POC) for digital twins, SCADA-integrated platforms, and AI-based monitoring systems for renewable energy applications.

**Project Overview
**The application is broken down into several functional modules:

**1. Data Simulation
**Since the goal is to demonstrate functionality in a standalone environment, the system starts by simulating realistic solar irradiance and ambient temperature readings. Energy output is calculated through a custom function that incorporates solar irradiance and penalizes output for temperatures that deviate from the optimal operating range (usually around 25°C for most PV panels).

Additionally, fault types are introduced in the dataset to replicate common real-world issues found in PV systems. These faults include: Shading, Soiling, Inverter Issues, Temperature Effects, Electrical Disconnects, and No Fault. This serves as the classification target for the fault detection model.

**2. Data Preprocessing
**The input features (solar irradiance and temperature) are normalized to bring them into the same scale. This helps improve the performance of machine learning models and also ensures a fair comparison of feature importances. Normalization is critical, especially when combining weather data and system-level outputs.

**3. Energy Output Regression Model
**A RandomForestRegressor model is trained to predict the energy output based on the normalized solar irradiance and temperature values. The dataset is split into training and testing sets to evaluate model accuracy. Performance is measured using two key metrics: Root Mean Squared Error (RMSE) and R² Score, which give a sense of how well the model generalizes to unseen data.

**4. Fault Classification Model
**To detect faults, a RandomForestClassifier is trained using the normalized weather features along with the predicted energy output. The goal of this model is to classify the system's condition into one of the predefined fault types. Class imbalance is addressed by using class_weight='balanced' to ensure that less frequent faults receive appropriate attention during training. The model is then evaluated using a classification report that includes precision, recall, and F1-score for each class.

**5. AI-Driven Fix Suggestion (NLP Integration)
**One of the most innovative features of this dashboard is the use of the DistilGPT2 language model from Hugging Face’s Transformers library. For every selected fault type, a descriptive prompt is generated to explain the problem and ask for actionable steps to resolve it. The model outputs a human-readable response containing suggestions. These are post-processed to extract meaningful instructions, such as repair recommendations or inspection procedures. This module effectively simulates how AI can assist in troubleshooting technical faults in solar systems.

**6. Interactive Dashboard
**The entire workflow is wrapped into a visually interactive dashboard using Dash and Plotly. Users can:

Visualize how energy output varies with solar irradiance or temperature.

View feature importances of the trained regression model.

Select different fault types from a dropdown menu to get instant AI-generated recommendations.

NOTE: The project will always be updated on time to time basis one of the next pssoible steps would be to experiemnt with the GPT model and better prompts to generate results 

The dashboard serves as both an educational and operational tool, allowing users to explore the relationship between weather conditions, system output, and technical faults in real-time.

