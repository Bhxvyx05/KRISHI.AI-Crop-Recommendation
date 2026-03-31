import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt

# --- 1. Caching for Performance ---
@st.cache_data
def load_data():
    """Loads the crop recommendation dataset."""
    try:
        df = pd.read_csv('Crop_recommendation.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'Crop_recommendation.csv' not found. Please make sure the dataset is in the same directory as 'app.py'.")
        return None

@st.cache_resource
def train_and_evaluate_models(df):
    """Prepares data, trains multiple models, identifies the best one, and returns it along with performance stats."""
    if df is None:
        return None, None, None, None, None

    X = df.drop('label', axis=1)
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1)
    }
    
    performance_data = []
    best_model_obj = None
    best_accuracy = 0.0
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        performance_data.append({"Model": name, "Accuracy": accuracy})
        trained_models[name] = model
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_obj = model
            
    performance_df = pd.DataFrame(performance_data).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    return best_model_obj, X, le, performance_df, trained_models, X_train

# --- 2. App Setup ---
st.set_page_config(page_title="Crop Recommendation App", layout="wide", initial_sidebar_state="expanded")

# --- Load Data and Train Models ---
df = load_data()

if df is not None:
    best_model, X, le, performance_df, trained_models, X_train_data = train_and_evaluate_models(df)

    # --- App Title and Description ---
    st.title("🌾 Crop Recommendation & Analysis Dashboard")
    st.write(
        "This app uses a Machine Learning model to recommend the best crop for your farm. "
        "Adjust the sliders in the sidebar to match your conditions, and the app will predict the optimal crop, "
        "explain its reasoning, and suggest alternatives."
    )

    # --- Sidebar for User Input ---
    st.sidebar.header("Enter Your Farm's Conditions:")
    def user_input_features():
        N = st.sidebar.slider('Nitrogen (N) Content (kg/ha)', int(df['N'].min()), int(df['N'].max()), 90)
        P = st.sidebar.slider('Phosphorus (P) Content (kg/ha)', int(df['P'].min()), int(df['P'].max()), 42)
        K = st.sidebar.slider('Potassium (K) Content (kg/ha)', int(df['K'].min()), int(df['K'].max()), 43)
        temperature = st.sidebar.slider('Temperature (°C)', 8.0, 44.0, 20.8, 0.1)
        humidity = st.sidebar.slider('Relative Humidity (%)', 14.0, 100.0, 82.0, 0.1)
        ph = st.sidebar.slider('Soil pH Value', 3.5, 9.9, 6.5, 0.1)
        rainfall = st.sidebar.slider('Rainfall (mm)', 20.0, 299.0, 202.9, 0.1)
        data = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # --- Main Panel ---
    col1, col2 = st.columns([2, 1.3])

    with col1:
        # --- 3. Prediction and Explanation ---
        st.header("📈 Prediction & Explanation")
        
        prediction_encoded = best_model.predict(input_df)
        prediction = le.inverse_transform(prediction_encoded)[0]
        prediction_proba = best_model.predict_proba(input_df)
        confidence = np.max(prediction_proba) * 100
        
        st.success(f"**Recommended Crop:** `{prediction.capitalize()}` (Confidence: {confidence:.2f}%)")

        st.subheader("Why this crop? (Explainable AI)")
        st.write("This plot shows feature contributions. **Red features** increase the likelihood of the prediction, while **blue features** decrease it.")

        # ###########################################################################
        # # FINAL CORRECTED SHAP PLOTTING LOGIC
        # ###########################################################################
        
        # Use the Explainer API with the LightGBM model
        explainer = shap.Explainer(trained_models["LightGBM"], X_train_data)
        
        # Calculate SHAP values for the user's single input
        shap_values_object = explainer(input_df)

        # Get the index of the predicted class
        class_index = list(best_model.classes_).index(prediction_encoded[0])
        
        # Let SHAP create its own plot, then we'll capture it.
        # We don't create fig, ax beforehand.
        shap.force_plot(
            base_value=shap_values_object.base_values[0, class_index],
            shap_values=shap_values_object.values[0, :, class_index],
            features=input_df,
            matplotlib=True,
            show=False,
            figsize=(10, 3)  # Adjusting size for better display
        )
        
        # Use plt.gcf() to get the current figure that SHAP created
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.clf() # Clear the figure to prevent it from being displayed on the next run
        # ###########################################################################

    with col2:
        # --- 4. Model Performance Comparison ---
        st.header("🏆 Model Performance Comparison")
        st.write("The app uses the model with the highest accuracy for predictions.")
        
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #4CAF50; color: white' if v else '' for v in is_max]
        
        st.dataframe(performance_df.style.apply(highlight_max, subset=['Accuracy']).format({'Accuracy': '{:.2%}'}), use_container_width=True)

    # --- 5. Counterfactual Analysis ---
    st.header("🤔 'What-If' Scenarios (Counterfactuals)")
    st.write("Discover what other crops might be viable if your farm's conditions were to change. Predictions below are based on the best-performing model.")

    def generate_counterfactuals(input_sample, model, le, feature_to_vary, value_range):
        original_prediction = le.inverse_transform(model.predict(input_sample))[0]
        alternatives = {}
        for value in value_range:
            temp_sample = input_sample.copy()
            temp_sample[feature_to_vary] = value
            new_prediction_encoded = model.predict(temp_sample)
            new_prediction = le.inverse_transform(new_prediction_encoded)[0]
            if new_prediction != original_prediction and new_prediction not in alternatives:
                change_direction = "increase" if value > input_sample[feature_to_vary].iloc[0] else "decrease"
                alternatives[new_prediction] = (value, change_direction)
        return alternatives

    cols = st.columns(3)
    with cols[0]:
        st.info("**💧 If Rainfall Changes...**")
        rain_range = np.linspace(df['rainfall'].min(), df['rainfall'].max(), 30)
        rain_counterfactuals = generate_counterfactuals(input_df, best_model, le, 'rainfall', rain_range)
        if rain_counterfactuals:
            for crop, (value, direction) in rain_counterfactuals.items():
                st.write(f"➡️ If rainfall were to **{direction}** to **{value:.0f} mm**, consider **{crop.capitalize()}**.")
        else:
            st.write("No simple alternatives found.")

    with cols[1]:
        st.warning("**🌿 If Potassium (K) Changes...**")
        k_range = np.linspace(df['K'].min(), df['K'].max(), 30)
        k_counterfactuals = generate_counterfactuals(input_df, best_model, le, 'K', k_range)
        if k_counterfactuals:
            for crop, (value, direction) in k_counterfactuals.items():
                st.write(f"➡️ If Potassium were to **{direction}** to **{value:.0f} kg/ha**, consider **{crop.capitalize()}**.")
        else:
            st.write("No alternatives found.")

    with cols[2]:
        st.error("**🌱 If Nitrogen (N) Changes...**")
        n_range = np.linspace(df['N'].min(), df['N'].max(), 30)
        n_counterfactuals = generate_counterfactuals(input_df, best_model, le, 'N', n_range)
        if n_counterfactuals:
            for crop, (value, direction) in n_counterfactuals.items():
                st.write(f"➡️ If Nitrogen were to **{direction}** to **{value:.0f} kg/ha**, consider **{crop.capitalize()}**.")
        else:
            st.write("No alternatives found.")