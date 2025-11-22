# app.py — Rewritten, robust Streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="E-Commerce Customer Intelligence", layout="wide")

# --- Config / artifact paths ---
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "best_lightgbm.pkl"
SCALER_PATH = ARTIFACT_DIR / "standard_scaler.pkl"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.pkl"
MODEL_DATA_PATH = ARTIFACT_DIR / "model_ready_dataset.csv"
RAW_DATA_PATH = "ecommerce_customer_churn_dataset.csv"


# --- Caching functions ---
@st.cache_data(ttl=3600)
def load_data():
    """Load raw and model-ready datasets (cached)."""
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load raw data from {RAW_DATA_PATH}: {e}")
        raw_df = pd.DataFrame()
    try:
        model_df = pd.read_csv(MODEL_DATA_PATH)
    except Exception as e:
        st.warning(f"Could not load model data from {MODEL_DATA_PATH}: {e}")
        model_df = pd.DataFrame()
    return raw_df, model_df


@st.cache_resource
def load_artifacts():
    """Load model, scaler and feature-list (cached resource)."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
    except Exception as e:
        st.error(f"Failed to load artifacts from {ARTIFACT_DIR}: {e}")
        raise
    # Ensure features is a list
    if isinstance(features, (pd.Series, np.ndarray)):
        features = list(features)
    return model, scaler, features


# --- Preprocessing helpers ---
def _build_base_vector(features):
    """
    Return a float Series of zeros indexed by provided features.
    Use float dtype to avoid integer/float incompatible assignments.
    """
    return pd.Series(0.0, index=list(features), dtype=float)


def _encode_categorical_into_base(base_vector, key, val):
    """
    For a categorical input key, set matching dummy column to 1
    e.g., key='Country', val='India' -> set 'Country_India' = 1 if exists.
    """
    # find columns that start with key + "_"
    prefix = f"{key}_"
    matches = [c for c in base_vector.index if c.startswith(prefix)]
    if not matches:
        return
    # set the exact match column to 1, others to 0
    target = f"{prefix}{val}"
    for c in matches:
        base_vector[c] = 1.0 if c == target else 0.0


def preprocess_inputs(customer_inputs: dict, features: list, scaler):
    """
    Build a single-row DataFrame matching scaler.feature_names_in_ (if present)
    and return a scaled numpy array ready for model.predict_proba / predict.
    This function is robust to missing engineered features (fills 0).
    """
    # Construct base vector with training features (float)
    base = _build_base_vector(features)

    # Fill numeric/continuous values directly if names match
    for k, v in customer_inputs.items():
        if k in base.index:
            # coerce numbers where possible
            try:
                base[k] = float(v)
            except Exception:
                # keep value as-is if cannot cast; set to 0 as safe fallback
                base[k] = 0.0
        else:
            # try categorical encoded columns like "Country_India"
            _encode_categorical_into_base(base, k, v)

    # If scaler has feature_names_in_, enforce exact schema (order + columns)
    if hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
        # reindex will drop extra base columns and fill missing with zeros
        base = base.reindex(expected, fill_value=0.0)
        X_df = pd.DataFrame([base.values], columns=expected)
    else:
        # Fall back: use provided features order
        X_df = pd.DataFrame([base.values], columns=base.index.tolist())

    # Scale using the fitted scaler — handle errors gracefully
    try:
        X_scaled = scaler.transform(X_df)
    except Exception as e:
        st.error(f"Scaler.transform failed: {e}")
        # Show debug info
        st.write("Input columns:", list(X_df.columns))
        if hasattr(scaler, "feature_names_in_"):
            st.write("Scaler expected:", list(scaler.feature_names_in_))
        raise

    return X_scaled  # shape (1, n_features)


# --- UI renderers ---
def render_eda(raw_df: pd.DataFrame):
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(raw_df):,}")
    churn_rate = raw_df["Churned"].mean() if "Churned" in raw_df.columns else np.nan
    c2.metric("Churn Rate", f"{churn_rate:.1%}" if not np.isnan(churn_rate) else "N/A")
    c3.metric("Avg Lifetime Value", f"${raw_df['Lifetime_Value'].mean():,.0f}" if "Lifetime_Value" in raw_df.columns else "N/A")

    st.write("Preview")
    st.dataframe(raw_df.head(25))

    st.markdown("#### Feature Distributions")
    numeric_opts = [c for c in raw_df.columns if pd.api.types.is_numeric_dtype(raw_df[c])]
    sample_features = ['Age', 'Membership_Years', 'Login_Frequency', 'Total_Purchases',
                       'Average_Order_Value', 'Lifetime_Value', 'Credit_Balance']
    # keep only those present
    plot_opts = [f for f in sample_features if f in raw_df.columns]
    if not plot_opts:
        st.info("No sample numeric columns found for histogram.")
    else:
        feature = st.selectbox("Select feature", options=plot_opts)
        fig = px.histogram(raw_df, x=feature, color='Churned' if 'Churned' in raw_df.columns else None,
                           nbins=40, title=f"{feature} Distribution by Churn")
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Missing Value Heatmap")
    missing_df = raw_df.isna()
    if missing_df.empty:
        st.info("No data to show heatmap.")
    else:
        fig = px.imshow(missing_df.astype(int), color_continuous_scale='Blues',
                        labels=dict(color='Missing (1=yes)'), aspect='auto')
        st.plotly_chart(fig, width='stretch')


def render_segments(model_df: pd.DataFrame):
    st.subheader("Customer Segments")
    if "Segment_Name" not in model_df.columns:
        st.info("Segment labels not found in saved dataset.")
        return

    segment_summary = model_df.groupby("Segment_Name").agg({
        "Churned": "mean",
        "Lifetime_Value": "mean",
        "Total_Purchases": "mean",
        "Discount_Usage_Rate": "mean",
        "Returns_Rate": "mean"
    }).rename(columns={"Churned": "Churn_Rate"})

    st.dataframe(segment_summary.style.format({
        "Churn_Rate": "{:.1%}",
        "Lifetime_Value": "${:,.0f}",
        "Total_Purchases": "{:.1f}",
        "Discount_Usage_Rate": "{:.1f}",
        "Returns_Rate": "{:.1f}"
    }))

    fig = px.bar(segment_summary.reset_index(),
                 x="Segment_Name", y="Churn_Rate",
                 title="Churn Rate by Segment",
                 color="Segment_Name",
                 labels={"Churn_Rate": "Churn Rate"})
    st.plotly_chart(fig, width='stretch')


def render_prediction(model, scaler, features, model_df, raw_df):
    st.subheader("Churn Prediction Sandbox")

    # --- Build input UI (only include columns we actually have in raw_df for sensible defaults) ---
    def med(col, default=0):
        return int(raw_df[col].median()) if col in raw_df.columns and pd.api.types.is_numeric_dtype(raw_df[col]) else default

    customer_inputs = {}
    customer_inputs["Age"] = st.slider("Age", 18, 80, med("Age", 35))
    customer_inputs["Membership_Years"] = st.slider("Membership Years", 0, 15, med("Membership_Years", 3))
    customer_inputs["Login_Frequency"] = st.slider("Monthly Login Frequency", 0, 40, med("Login_Frequency", 12))
    customer_inputs["Session_Duration_Avg"] = st.slider("Avg Session Duration (min)", 0.0, 60.0, float(med("Session_Duration_Avg", 25)))
    customer_inputs["Pages_Per_Session"] = st.slider("Pages per Session", 1.0, 25.0, float(med("Pages_Per_Session", 8)))
    customer_inputs["Cart_Abandonment_Rate"] = st.slider("Cart Abandonment Rate (%)", 0.0, 100.0, 45.0)
    customer_inputs["Wishlist_Items"] = st.slider("Wishlist Items", 0, 20, 3)
    customer_inputs["Total_Purchases"] = st.slider("Purchases (12m)", 0.0, 50.0, float(med("Total_Purchases", 12)))
    customer_inputs["Average_Order_Value"] = st.slider("Average Order Value", 10.0, 300.0, float(med("Average_Order_Value", 120)))
    customer_inputs["Days_Since_Last_Purchase"] = st.slider("Days Since Last Purchase", 0, 365, med("Days_Since_Last_Purchase", 40))
    customer_inputs["Discount_Usage_Rate"] = st.slider("Discount Usage Rate (%)", 0.0, 100.0, float(med("Discount_Usage_Rate", 35)))
    customer_inputs["Returns_Rate"] = st.slider("Returns Rate (%)", 0.0, 100.0, float(med("Returns_Rate", 8)))
    customer_inputs["Email_Open_Rate"] = st.slider("Email Open Rate (%)", 0.0, 100.0, float(med("Email_Open_Rate", 25)))
    customer_inputs["Customer_Service_Calls"] = st.slider("Customer Service Calls", 0, 20, med("Customer_Service_Calls", 2))
    customer_inputs["Product_Reviews_Written"] = st.slider("Product Reviews Written", 0, 20, med("Product_Reviews_Written", 1))
    customer_inputs["Social_Media_Engagement_Score"] = st.slider("Social Media Score", 0.0, 100.0, float(med("Social_Media_Engagement_Score", 30)))
    customer_inputs["Mobile_App_Usage"] = st.slider("Mobile App Usage (hrs/month)", 0.0, 80.0, float(med("Mobile_App_Usage", 20)))
    customer_inputs["Payment_Method_Diversity"] = st.slider("Payment Method Diversity", 1, 6, med("Payment_Method_Diversity", 2))
    customer_inputs["Lifetime_Value"] = st.slider("Lifetime Value", 100.0, 6000.0, float(med("Lifetime_Value", 1800)))
    customer_inputs["Credit_Balance"] = st.slider("Credit Balance (loyalty points)", 0.0, 6000.0, float(med("Credit_Balance", 1000)))

    # Categorical selects with defaults
    country_options = sorted(raw_df["Country"].dropna().unique()) if "Country" in raw_df.columns else ["Unknown"]
    city_options = sorted(raw_df["City"].dropna().unique()) if "City" in raw_df.columns else ["Unknown"]
    signup_quarters = raw_df["Signup_Quarter"].dropna().unique() if "Signup_Quarter" in raw_df.columns else ["Q1", "Q2", "Q3", "Q4"]

    customer_inputs["Country"] = st.selectbox("Country", country_options)
    customer_inputs["City"] = st.selectbox("City", city_options)
    customer_inputs["Signup_Quarter"] = st.selectbox("Signup Quarter", signup_quarters)
    customer_inputs["Membership_Stage"] = st.selectbox("Membership Stage", ["Newbie", "Growing", "Established", "Veteran"])
    customer_inputs["Seasonal_Activity"] = st.selectbox("Seasonal Activity", ["High", "Moderate", "Peak", "Low"])

    # --- Engineered features (safe computations) ---
    try:
        customer_inputs["Recency"] = 365 - float(customer_inputs["Days_Since_Last_Purchase"])
    except Exception:
        customer_inputs["Recency"] = 0.0
    customer_inputs["Frequency"] = float(customer_inputs["Total_Purchases"]) / (float(customer_inputs["Membership_Years"]) + 1e-3)
    customer_inputs["Monetary"] = float(customer_inputs["Average_Order_Value"]) * float(customer_inputs["Total_Purchases"])
    customer_inputs["Engagement_Index"] = np.cbrt(
        float(customer_inputs["Login_Frequency"]) * max(float(customer_inputs["Session_Duration_Avg"]), 1e-3) *
        max(float(customer_inputs["Pages_Per_Session"]), 1e-3)
    )
    customer_inputs["Support_Intensity"] = float(customer_inputs["Customer_Service_Calls"]) / (float(customer_inputs["Total_Purchases"]) + 1)
    customer_inputs["Purchase_to_Login_Ratio"] = float(customer_inputs["Total_Purchases"]) / (float(customer_inputs["Login_Frequency"]) + 1)
    customer_inputs["Discount_to_Order_Value"] = float(customer_inputs["Discount_Usage_Rate"]) / (float(customer_inputs["Average_Order_Value"]) + 1)
    customer_inputs["Return_to_Purchase_Ratio"] = float(customer_inputs["Returns_Rate"]) / (float(customer_inputs["Total_Purchases"]) + 1)
    try:
        customer_inputs["Signup_Quarter_Num"] = int(str(customer_inputs["Signup_Quarter"]).replace("Q", ""))
    except Exception:
        customer_inputs["Signup_Quarter_Num"] = 1
    customer_inputs["Lifetime_Value_Squared"] = float(customer_inputs["Lifetime_Value"]) ** 2
    customer_inputs["High_Returns_Flag"] = int(float(customer_inputs["Returns_Rate"]) > (raw_df["Returns_Rate"].median() * 1.5) if "Returns_Rate" in raw_df.columns else 0)
    customer_inputs["Cross_Channel_Engagement"] = (
        float(customer_inputs["Social_Media_Engagement_Score"]) * float(customer_inputs["Mobile_App_Usage"])
    )

    # --- Preprocess and predict ---
    try:
        X_scaled = preprocess_inputs(customer_inputs, features, scaler)  # shape (1, n)
        pred_proba = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else model.predict(X_scaled)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    st.metric("Predicted Churn Probability", f"{pred_proba:.1%}")
    threshold = st.slider("Decision Threshold (cost-sensitive)", 0.1, 0.9, 0.5, 0.05)
    prediction = "Churn" if pred_proba >= threshold else "Retain"
    st.success(f"Recommendation: **{prediction}** (threshold={threshold:.2f})")
    st.caption("Cost model: $500 loss for missed churners, $50 cost for unnecessary retention campaigns.")


def render_cost_analysis(model, scaler, features, model_df):
    st.subheader("Cost-Sensitive Optimization")

    # If model_df is empty, warn and return
    if model_df.empty:
        st.warning("Model dataset is not available for cost analysis.")
        return

    # Align model_df columns with scaler expectation (drop or fill as needed)
    if hasattr(scaler, "feature_names_in_"):
        X = model_df.reindex(columns=scaler.feature_names_in_, fill_value=0.0)
    else:
        X = model_df.reindex(columns=features, fill_value=0.0)

    # Ensure y exists
    if "Churned" not in model_df.columns:
        st.warning("Model dataset does not include 'Churned' target.")
        return
    y = model_df["Churned"].astype(int)

    # Safe transform & predict
    try:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_scaled)
    except Exception as e:
        st.error(f"Failed to compute probabilities: {e}")
        st.write("X columns:", list(X.columns))
        if hasattr(scaler, "feature_names_in_"):
            st.write("Scaler expected:", list(scaler.feature_names_in_))
        return

    thresholds = np.linspace(0.1, 0.9, 25)
    cost_fn = st.number_input("Cost of False Negative ($)", value=500, min_value=0, max_value=10000, step=50)
    cost_fp = st.number_input("Cost of False Positive ($)", value=50, min_value=0, max_value=10000, step=10)

    expected_costs = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        cost = fn * cost_fn + fp * cost_fp
        expected_costs.append(cost)

    df_cost = pd.DataFrame({"threshold": thresholds, "expected_cost": expected_costs})
    best_row = df_cost.loc[df_cost["expected_cost"].idxmin()]

    fig = px.line(df_cost, x="threshold", y="expected_cost", title="Expected Loss vs Threshold")
    fig.add_vline(x=best_row["threshold"], line_dash="dash", line_color="red")
    st.plotly_chart(fig, width='stretch')
    st.metric("Optimal Threshold", f"{best_row['threshold']:.2f}")
    st.metric("Minimum Expected Cost", f"${best_row['expected_cost']:,.0f}")


# --- Main app execution ---
def main():
    # Load data & artifacts
    raw_df, model_df = load_data()
    try:
        model, scaler, features = load_artifacts()
    except Exception:
        st.stop()

    st.title("Customer Segmentation & Churn Intelligence")
    st.write(
        "Explore segments, diagnose churn drivers, and run what-if simulations powered "
        "by the trained LightGBM ensemble."
    )

    tabs = st.tabs(["EDA", "Segments", "Prediction", "Cost & ROI"])

    with tabs[0]:
        render_eda(raw_df)

    with tabs[1]:
        render_segments(model_df)

    with tabs[2]:
        render_prediction(model, scaler, features, model_df, raw_df)

    with tabs[3]:
        render_cost_analysis(model, scaler, features, model_df)


if __name__ == "__main__":
    main()



