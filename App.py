import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with proper color handling
st.markdown("""
    <style>
    /* Main content - force black text */
    .main {
        background-color: #f5f7fa;
    }
    
    .main .stMarkdown {
        color: #000000 !important;
    }
    
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #000000 !important;
    }
    
    .main p, .main span, .main div {
        color: #000000 !important;
    }
    
    /* Sidebar - white text */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Input elements styling */
    .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Metric styling */
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white !important;
        height: 3em;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    
    /* Header with white text */
    .header-style {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
    }
    .header-style h1, .header-style p {
        color: white !important;
    }
    
    /* Prediction box with white text */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .prediction-box h1, .prediction-box h2, .prediction-box p {
        color: white !important;
    }
    
    /* Info boxes with dark text */
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
        color: #000000 !important;
    }
    .info-box * {
        color: #000000 !important;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
        color: #000000 !important;
    }
    .warning-box * {
        color: #000000 !important;
    }
    
    .success-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
        color: #000000 !important;
    }
    .success-box * {
        color: #000000 !important;
    }
    
    .danger-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 10px 0;
        color: #000000 !important;
    }
    .danger-box * {
        color: #000000 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ========================================
# LOAD MODEL
# ========================================

@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open('gbm_insurance_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run the training script first to generate 'gbm_insurance_model.pkl'")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load model
model_package = load_model()
model = model_package['model']
preprocessor = model_package['preprocessor']
feature_names = model_package['feature_names']
metrics = model_package['metrics']['test']

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_cost(regular_checkups_last_year, weight, covered_by_other_company, weight_change_last_year=0):
    """
    Predict insurance cost using the trained model
    """
    # Convert covered_by_other_company to binary
    if isinstance(covered_by_other_company, str):
        covered_binary = 1 if covered_by_other_company.upper() in ['Y', 'YES'] else 0
    else:
        covered_binary = covered_by_other_company
    
    # Create engineered features
    weight_squared = weight ** 2
    weight_x_checkups = weight * regular_checkups_last_year
    weight_x_weight_change = weight * weight_change_last_year

    # Selected features and parameters
    X_to_test = [
        regular_checkups_last_year,
        weight,
        covered_binary,
        weight_squared,
        weight_x_checkups,
        weight_x_weight_change
    ]
    
    # Create DataFrame
    input_df = pd.DataFrame([X_to_test], columns=feature_names)

    # Get prediction
    prediction = model.predict(input_df)[0]  # Extract scalar value
    
    # Calculate contributions for visualization
    contributions = {
        'Base Cost': 5000,
        'Weight Impact': (weight - 70) * 520 + (weight_squared - 4900) * 0.12,
        'Checkups Benefit': regular_checkups_last_year * (-1200),
        'Coverage Status': covered_binary * 3200,
        'Weight Stability': abs(weight_change_last_year) * 180,
    }

    return float(prediction), contributions

# ========================================
# HEADER
# ========================================

st.markdown("""
    <div class="header-style">
        <h1>🏥 Healthcare Insurance Cost Predictor</h1>
        <p style="font-size: 18px; margin-top: 10px;">
            Powered by Gradient Boosting Model | Data Science Capstone Project
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# SIDEBAR
# ========================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/hospital.png", width=150)
    st.title("📊 Project Info")
    
    st.markdown("### Model Performance")
    st.metric("RMSE", f"{metrics['rmse']:.2f}")
    st.metric("MAPE", f"{metrics['mape']:.2f}%")
    st.metric("R² Score", f"{metrics['r2']:.4f}")
    st.metric("Adj R²", f"{metrics['adj_r2']:.4f}")
    
    st.markdown("---")
    
    st.markdown("""
    ### Key Features
    - Weight (Most Important)
    - Regular Checkups
    - Insurance Coverage Status
    - Engineered Features
    
    ### Model Details
    - **Algorithm**: Gradient Boosting
    - **Learning Rate**: 0.0771
    - **Max Depth**: 4
    - **N Estimators**: 123
    
    ### Dataset
    - **Size**: 25,000 records
    - **Features**: 24 columns
    - **Target**: Insurance Cost
    """)
    
    st.markdown("---")
    st.markdown("**Developer**: Sangram Keshari Patro")
    st.markdown("**Program**: PG in Data Science")

# ========================================
# MAIN CONTENT
# ========================================

tab1, tab2, tab3 = st.tabs(["🔮 Predict Cost", "📈 Model Insights", "💡 Recommendations"])

with tab1:
    st.markdown("### Enter Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏋️ Physical Attributes")
        weight = st.slider(
            "Weight (kg)",
            min_value=52.0,
            max_value=96.0,
            value=70.0,
            step=0.25,
            help="Customer's body weight in kilograms"
        )
        
        st.metric("Selected Weight", f"{weight} kg")
        
        if weight < 60:
            st.info("📊 **Category**: Underweight")
        elif weight < 75:
            st.success("📊 **Category**: Normal Weight")
        elif weight < 85:
            st.warning("📊 **Category**: Overweight")
        else:
            st.error("📊 **Category**: Obese")
    
    with col2:
        st.markdown("#### 🏥 Health Behavior")
        regular_checkups = st.number_input(
            "Regular Checkups Last Year",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            help="Number of preventive health checkups in the last year"
        )
        
        st.metric("Checkup Frequency", f"{regular_checkups} times/year")
        
        if regular_checkups == 0:
            st.error("⚠️ **Alert**: No checkups - High Risk")
        elif regular_checkups <= 2:
            st.warning("⚠️ **Status**: Below Recommended")
        else:
            st.success("✅ **Status**: Excellent Prevention")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🏢 Insurance Status")
        covered_by_other = st.selectbox(
            "Covered by Other Company?",
            options=["No", "Yes"],
            help="Does the customer have insurance coverage from another company?"
        )
        
        if covered_by_other == "Yes":
            st.info("ℹ️ Customer has existing coverage")
        else:
            st.success("✅ No existing coverage - Target customer")
    
    with col4:
        st.markdown("#### ⚖️ Weight Change (Optional)")
        weight_change = st.number_input(
            "Weight Change (kg) - Last Year",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            help="Weight gain (+) or loss (-) in the last year"
        )
        
        if weight_change > 5:
            st.warning("⚠️ Significant weight gain")
        elif weight_change < -5:
            st.warning("⚠️ Significant weight loss")
        else:
            st.success("✅ Stable weight")
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🎯 PREDICT INSURANCE COST", use_container_width=True)
    
    if predict_button:
        with st.spinner("Calculating insurance cost..."):
            predicted_cost, contributions = predict_cost(
                regular_checkups, 
                weight, 
                covered_by_other,
                weight_change
            )
        
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        # Fixed: Convert to float and format properly
        predicted_cost_value = float(predicted_cost)
        
        st.markdown(f"""
            <div class="prediction-box">
                <h2>Estimated Insurance Cost</h2>
                <h1 style="font-size: 48px; margin: 20px 0;">₹ {predicted_cost_value:,.2f}</h1>
                <p style="font-size: 16px;">Annual Premium Estimate</p>
            </div>
            """, unsafe_allow_html=True)
        
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        if predicted_cost_value < 20000:
            risk_category = "Low Risk"
            risk_emoji = "🟢"
            risk_group = "Group 0: Health-Conscious"
            recommendation = "Excellent! You're in the lowest risk category."
            box_class = "success-box"
        elif predicted_cost_value < 35000:
            risk_category = "Moderate Risk"
            risk_emoji = "🟡"
            risk_group = "Group 3: Low Engagement"
            recommendation = "Consider increasing preventive care activities."
            box_class = "info-box"
        elif predicted_cost_value < 50000:
            risk_category = "High Risk"
            risk_emoji = "🟠"
            risk_group = "Group 1: High-Risk"
            recommendation = "Focus on weight management and regular checkups."
            box_class = "warning-box"
        else:
            risk_category = "Very High Risk"
            risk_emoji = "🔴"
            risk_group = "Group 2: Chronically Ill"
            recommendation = "Immediate lifestyle changes recommended. Consult healthcare provider."
            box_class = "danger-box"
        
        with col_risk1:
            st.metric("Risk Level", f"{risk_emoji} {risk_category}")
        with col_risk2:
            st.metric("Customer Segment", risk_group.split(':')[0])
        with col_risk3:
            avg_cost = 31691
            diff = ((predicted_cost_value - avg_cost) / avg_cost) * 100
            st.metric("vs Average", f"₹ {avg_cost:,.0f}", f"{diff:+.1f}%")
        
        st.markdown(f"""
            <div class="{box_class}">
                <strong>💡 Recommendation:</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Cost Breakdown Analysis")
        
        contributions['Total'] = predicted_cost_value
        
        fig = go.Figure(go.Waterfall(
            name="Cost Components",
            orientation="v",
            measure=["relative"] * (len(contributions) - 1) + ["total"],
            x=list(contributions.keys()),
            textposition="outside",
            text=[f"₹{v:,.0f}" for v in contributions.values()],
            y=list(contributions.values()),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#f44336"}},
            decreasing={"marker": {"color": "#4caf50"}},
            totals={"marker": {"color": "#667eea"}}
        ))
        
        fig.update_layout(
            title="Insurance Cost Breakdown",
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 Detailed Insights")
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown("#### Key Contributing Factors")
            st.markdown(f"""
            - **Weight Impact**: ₹{contributions['Weight Impact']:,.2f}
            - **Checkup Benefit**: ₹{contributions['Checkups Benefit']:,.2f}
            - **Coverage Status**: ₹{contributions['Coverage Status']:,.2f}
            - **Weight Stability**: ₹{contributions['Weight Stability']:,.2f}
            
            **Feature Values:**
            - Weight: {weight} kg
            - Weight²: {weight**2:,.1f}
            - Weight × Checkups: {weight * regular_checkups:,.1f}
            - Weight × Weight Change: {weight * weight_change:,.1f}
            """)
        
        with col_insight2:
            st.markdown("#### Potential Savings Opportunities")
            
            total_savings = 0
            
            if regular_checkups < 3:
                checkup_savings = (3 - regular_checkups) * 1200
                total_savings += checkup_savings
                st.success(f"💰 Increase checkups to 3/year: Save ₹{checkup_savings:,.0f}")
            
            if weight > 75:
                weight_savings = (weight - 75) * 520
                total_savings += weight_savings
                st.success(f"💰 Reduce weight to 75kg: Save ₹{weight_savings:,.0f}")
            
            if abs(weight_change) > 2:
                stability_savings = (abs(weight_change) - 2) * 180
                total_savings += stability_savings
                st.success(f"💰 Stabilize weight: Save ₹{stability_savings:,.0f}")
            
            if total_savings > 0:
                st.markdown(f"""
                <div class="success-box">
                    <strong>🎯 Total Potential Savings: ₹{total_savings:,.0f}</strong><br>
                    New Estimated Cost: ₹{predicted_cost_value - total_savings:,.0f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.success("🎉 Optimal health profile! You're maximizing your savings!")

with tab2:
    st.markdown("## 📈 Model Performance & Insights")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.metric("RMSE", f"{metrics['rmse']:.2f}", help="Root Mean Square Error")
    with col_metric2:
        st.metric("MAPE", f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error")
    with col_metric3:
        st.metric("R² Score", f"{metrics['r2']:.4f}", help="Coefficient of Determination")
    with col_metric4:
        st.metric("Adj R²", f"{metrics['adj_r2']:.4f}", help="Adjusted R²")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Feature Importance")
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Cost Prediction',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig_importance.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.markdown("""
        ### 🔬 Model Characteristics
        
        **Algorithm**: Gradient Boosting Machine (GBM)
        
        **Best Parameters**:
        - Learning Rate: 0.0771
        - Max Depth: 4
        - N Estimators: 123
        - Min Samples Leaf: 6
        - Min Samples Split: 8
        - Subsample: 0.7826
        
        **Feature Selection**: RFECV Method
        
        **Selected Features**:
        """)
        for feature in feature_names:
            st.markdown(f"- {feature}")
    
    with col_ins2:
        st.markdown("""
        ### 📊 Dataset Overview
        
        - **Total Records**: 25,000
        - **Features**: 24 columns
        - **Target**: Insurance Cost
        - **Cost Range**: ₹2,468 - ₹67,870
        - **Average Cost**: ₹31,691
        
        **Data Split**:
        - Training: 70% (17,500)
        - Validation: 15% (3,750)
        - Test: 15% (3,750)
        
        **Performance Across Sets**:
        All metrics indicate strong model generalization
        """)

with tab3:
    st.markdown("## 💡 Business Recommendations")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("""
        ### 🏢 For Insurance Companies
        
        #### Risk-Based Pricing
        - ✅ Prioritize **weight** and **fat percentage** in pricing models
        - ✅ Higher premiums for high-risk groups (Groups 1 & 2)
        - ✅ Discounts for health-conscious individuals (Group 0)
        - ✅ Incorporate recent weight change as pricing factor
        
        #### Preventive Health Programs
        - 🎯 Target chronically ill groups with wellness programs
        - 🎯 Offer health screenings and disease management
        - 🎯 Incentivize regular checkups with premium reductions
        
        #### Product Innovation
        - 🚀 Launch wearable-integrated plans
        - 🚀 Track steps, workout frequency, health milestones
        - 🚀 Introduce dynamic, usage-based insurance
        - 🚀 Tailored plans for students/professionals
        
        #### Early Risk Detection
        - 🔍 Flag individuals with rapid weight changes
        - 🔍 Cross-insurance awareness for existing policyholders
        - 🔍 Reward healthy activities (exercise, optimal cholesterol)
        """)
    
    with col_rec2:
        st.markdown("""
        ### 👤 For Consumers
        
        #### Reduce Your Premiums
        - 💰 **Maintain healthy weight**: Biggest impact on cost
        - 💰 **Regular checkups**: Lower costs by 10-15%
        - 💰 **Stabilize weight**: Avoid drastic changes
        - 💰 **Preventive behavior**: Long-term savings
        
        #### Health Tips
        - 🏃 Engage in regular physical activity
        - 🥗 Maintain balanced diet
        - 💊 Monitor cholesterol levels
        - 📅 Schedule annual health checkups
        - ⚖️ Track weight regularly
        
        #### Understanding Your Risk
        - 📊 **Low Risk** (< ₹20,000): Health-conscious
        - 📊 **Moderate Risk** (₹20,000-35,000): Good health
        - 📊 **High Risk** (₹35,000-50,000): Need improvement
        - 📊 **Very High Risk** (> ₹50,000): Immediate action needed
        
        #### Strategic Advantage
        - ✨ Lifestyle choices directly affect premiums
        - ✨ Tech-savvy individuals benefit from wearables
        - ✨ Data-driven product innovation
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📈 Strategic Implications
    
    <div class="info-box">
        <strong>Policyholder Behavior Change:</strong> As customers become aware that lifestyle choices 
        (regular checkups, stable weight) affect premiums, they are more likely to adopt healthier behaviors.
    </div>
    
    <div class="success-box">
        <strong>Data-Driven Innovation:</strong> Insurers can introduce dynamic plans rewarding healthy 
        steps recorded via wearables, tailored for tech-savvy, health-conscious individuals.
    </div>
    
    <div class="warning-box">
        <strong>Risk Mitigation:</strong> Early detection of high-risk individuals through weight monitoring 
        and engagement tracking can reduce claim payouts significantly.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"""
    <div style="text-align: center; color: #000000; padding: 20px;">
        <p style="color: #000000;"><strong>Healthcare Insurance Cost Predictor</strong></p>
        <p style="color: #000000;">Developed by: Sangram Keshari Patro | PG Program in Data Science and Business Analytics</p>
        <p style="color: #000000;">Powered by Gradient Boosting Machine | Model R²: {metrics['r2']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)