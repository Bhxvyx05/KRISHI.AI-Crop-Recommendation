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


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KRISHI.AI - Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    """Load the crop recommendation dataset."""

    try:
        df = pd.read_csv("Crop_recommendation.csv")
        return df

    except FileNotFoundError:
        st.error(
            "❌ Crop_recommendation.csv was not found. "
            "Please keep the CSV file in the same folder as app.py."
        )
        return None


# ============================================================
# 3. TRAIN AND EVALUATE MODELS
# ============================================================

@st.cache_resource
def train_and_evaluate_models(df):
    """
    Prepare data, train multiple ML models,
    compare their performance, and return the best model.
    """

    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    # Models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(
            random_state=42,
            verbosity=-1
        )
    }

    performance_data = []
    trained_models = {}

    best_model = None
    best_accuracy = -1

    # Train and evaluate
    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        performance_data.append({
            "Model": name,
            "Accuracy": accuracy
        })

        trained_models[name] = model

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    performance_df = (
        pd.DataFrame(performance_data)
        .sort_values(
            by="Accuracy",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        best_model,
        X,
        le,
        performance_df,
        trained_models,
        X_train
    )


# ============================================================
# 4. LOAD DATA AND TRAIN MODELS
# ============================================================

df = load_data()


if df is not None:

    (
        best_model,
        X,
        le,
        performance_df,
        trained_models,
        X_train_data
    ) = train_and_evaluate_models(df)


    # ========================================================
    # 5. APPLICATION HEADER
    # ========================================================

    st.title("🌾 KRISHI.AI")
    st.subheader("AI-Powered Crop Recommendation & Analysis")

    st.write(
        "KRISHI.AI uses machine learning to recommend the most suitable "
        "crop based on soil and environmental conditions."
    )


    # ========================================================
    # 6. SIDEBAR USER INPUT
    # ========================================================

    st.sidebar.header("🌱 Enter Farm Conditions")

    N = st.sidebar.slider(
        "Nitrogen (N) Content (kg/ha)",
        int(df["N"].min()),
        int(df["N"].max()),
        int(df["N"].median())
    )

    P = st.sidebar.slider(
        "Phosphorus (P) Content (kg/ha)",
        int(df["P"].min()),
        int(df["P"].max()),
        int(df["P"].median())
    )

    K = st.sidebar.slider(
        "Potassium (K) Content (kg/ha)",
        int(df["K"].min()),
        int(df["K"].max()),
        int(df["K"].median())
    )

    temperature = st.sidebar.slider(
        "Temperature (°C)",
        float(df["temperature"].min()),
        float(df["temperature"].max()),
        float(df["temperature"].median()),
        0.1
    )

    humidity = st.sidebar.slider(
        "Relative Humidity (%)",
        float(df["humidity"].min()),
        float(df["humidity"].max()),
        float(df["humidity"].median()),
        0.1
    )

    ph = st.sidebar.slider(
        "Soil pH Value",
        float(df["ph"].min()),
        float(df["ph"].max()),
        float(df["ph"].median()),
        0.1
    )

    rainfall = st.sidebar.slider(
        "Rainfall (mm)",
        float(df["rainfall"].min()),
        float(df["rainfall"].max()),
        float(df["rainfall"].median()),
        0.1
    )


    # Create model input
    input_df = pd.DataFrame({
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall]
    })


    # ========================================================
    # 7. MAIN DASHBOARD
    # ========================================================

    col1, col2 = st.columns([2, 1.3])


    # ========================================================
    # COLUMN 1 - PREDICTION + SHAP
    # ========================================================

    with col1:

        st.header("📈 Prediction & Explanation")

        # Prediction
        prediction_encoded = best_model.predict(input_df)

        prediction = le.inverse_transform(
            prediction_encoded.astype(int)
        )[0]

        # Prediction probability
        prediction_proba = best_model.predict_proba(input_df)

        confidence = np.max(prediction_proba) * 100

        st.success(
            f"🌾 **Recommended Crop:** "
            f"`{prediction.capitalize()}`  \n"
            f"**Confidence:** {confidence:.2f}%"
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.subheader("🔍 Why this crop? — Explainable AI")

        st.write(
            "The SHAP explanation shows how each feature contributes "
            "to the prediction."
        )

        try:

            # Use the LightGBM model specifically for SHAP
            lightgbm_model = trained_models["LightGBM"]

            explainer = shap.TreeExplainer(
                lightgbm_model
            )

            shap_values = explainer.shap_values(
                input_df
            )

            # Handle multiclass SHAP output
            if isinstance(shap_values, list):

                class_index = int(
                    prediction_encoded[0]
                )

                shap_values_for_class = shap_values[
                    class_index
                ]

            else:

                # Newer SHAP versions can return
                # a 3D array for multiclass models
                if len(shap_values.shape) == 3:

                    class_index = int(
                        prediction_encoded[0]
                    )

                    shap_values_for_class = shap_values[
                        0, :, class_index
                    ]

                else:

                    shap_values_for_class = shap_values[
                        0
                    ]

            # Create SHAP bar plot
            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            feature_importance = pd.Series(
                np.abs(shap_values_for_class),
                index=input_df.columns
            ).sort_values()

            feature_importance.plot(
                kind="barh",
                ax=ax
            )

            ax.set_title(
                "Feature Contribution to Prediction"
            )

            ax.set_xlabel(
                "Absolute SHAP Value"
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        except Exception as e:

            st.warning(
                f"SHAP explanation could not be generated: {e}"
            )


    # ========================================================
    # COLUMN 2 - MODEL PERFORMANCE
    # ========================================================

    with col2:

        st.header("🏆 Model Performance")

        st.write(
            "The model with the highest test accuracy "
            "is selected for crop prediction."
        )

        def highlight_max(s):

            is_max = s == s.max()

            return [
                "background-color: #4CAF50; color: white"
                if value
                else ""
                for value in is_max
            ]

        styled_performance = (
            performance_df.style
            .apply(
                highlight_max,
                subset=["Accuracy"]
            )
            .format({
                "Accuracy": "{:.2%}"
            })
        )

        st.dataframe(
            styled_performance,
            use_container_width=True
        )

        # Best model
        best_model_name = performance_df.iloc[0]["Model"]

        st.info(
            f"🥇 **Best Performing Model:** "
            f"{best_model_name}"
        )


    # ========================================================
    # 8. COUNTERFACTUAL ANALYSIS
    # ========================================================

    st.header("🤔 What-If Scenarios")

    st.write(
        "Explore how changing individual farm conditions "
        "can influence the recommended crop."
    )


    def generate_counterfactuals(
        input_sample,
        model,
        le,
        feature_to_vary,
        value_range
    ):
        """
        Change one feature at a time and observe
        whether the predicted crop changes.

        A float copy is used to avoid pandas dtype
        errors when testing decimal counterfactual values.
        """

        # IMPORTANT:
        # Convert temporary dataframe to float so that
        # np.linspace decimal values can be assigned safely.
        base_sample = input_sample.copy().astype(float)

        original_prediction_encoded = model.predict(
            base_sample
        )

        original_prediction = le.inverse_transform(
            original_prediction_encoded.astype(int)
        )[0]

        alternatives = {}

        for value in value_range:

            temp_sample = base_sample.copy()

            # Safe assignment
            temp_sample.loc[
                temp_sample.index[0],
                feature_to_vary
            ] = float(value)

            new_prediction_encoded = model.predict(
                temp_sample
            )

            new_prediction = le.inverse_transform(
                new_prediction_encoded.astype(int)
            )[0]

            if (
                new_prediction != original_prediction
                and new_prediction not in alternatives
            ):

                original_value = float(
                    input_sample[
                        feature_to_vary
                    ].iloc[0]
                )

                if value > original_value:
                    change_direction = "increase"
                else:
                    change_direction = "decrease"

                alternatives[new_prediction] = (
                    value,
                    change_direction
                )

        return alternatives


    # ========================================================
    # COUNTERFACTUAL COLUMNS
    # ========================================================

    cols = st.columns(3)


    # ========================================================
    # RAINFALL
    # ========================================================

    with cols[0]:

        st.info(
            "💧 **If Rainfall Changes...**"
        )

        rain_range = np.linspace(
            df["rainfall"].min(),
            df["rainfall"].max(),
            30
        )

        rain_counterfactuals = generate_counterfactuals(
            input_df,
            best_model,
            le,
            "rainfall",
            rain_range
        )

        if rain_counterfactuals:

            for crop, (
                value,
                direction
            ) in rain_counterfactuals.items():

                st.write(
                    f"➡️ If rainfall were to "
                    f"**{direction}** to "
                    f"**{value:.0f} mm**, "
                    f"consider **{crop.capitalize()}**."
                )

        else:

            st.write(
                "No simple alternatives found."
            )


    # ========================================================
    # POTASSIUM
    # ========================================================

    with cols[1]:

        st.warning(
            "🌿 **If Potassium (K) Changes...**"
        )

        k_range = np.linspace(
            df["K"].min(),
            df["K"].max(),
            30
        )

        k_counterfactuals = generate_counterfactuals(
            input_df,
            best_model,
            le,
            "K",
            k_range
        )

        if k_counterfactuals:

            for crop, (
                value,
                direction
            ) in k_counterfactuals.items():

                st.write(
                    f"➡️ If Potassium were to "
                    f"**{direction}** to "
                    f"**{value:.0f} kg/ha**, "
                    f"consider **{crop.capitalize()}**."
                )

        else:

            st.write(
                "No alternatives found."
            )


    # ========================================================
    # NITROGEN
    # ========================================================

    with cols[2]:

        st.error(
            "🌱 **If Nitrogen (N) Changes...**"
        )

        n_range = np.linspace(
            df["N"].min(),
            df["N"].max(),
            30
        )

        n_counterfactuals = generate_counterfactuals(
            input_df,
            best_model,
            le,
            "N",
            n_range
        )

        if n_counterfactuals:

            for crop, (
                value,
                direction
            ) in n_counterfactuals.items():

                st.write(
                    f"➡️ If Nitrogen were to "
                    f"**{direction}** to "
                    f"**{value:.0f} kg/ha**, "
                    f"consider **{crop.capitalize()}**."
                )

        else:

            st.write(
                "No alternatives found."
            )


    # ========================================================
    # 9. CURRENT INPUT SUMMARY
    # ========================================================

    st.header("📋 Current Farm Conditions")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Please make sure Crop_recommendation.csv "
        "is present in the project directory."
    )