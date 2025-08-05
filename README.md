# Insurance-Cost-Estimation-Healthcare-Analytics-Capstone-Project
**Author:** Sangram Keshari Patro
Go this this link for the presentation of this project:-  https://healthcare-capstone-proj-o60ijkm.gamma.site/

## 📌 Project Summary

Built a predictive model to estimate insurance costs using health, lifestyle, and demographic data from 25,000+ individuals. The goal was to support fair policy pricing, promote preventive care, and reduce insurer risk.

## 🧠 Key Highlights

- Performed EDA and clustering (K-Prototypes) to identify high-risk segments.
- Engineered features (e.g., weight × checkups, log(weight), weight²) for better model performance.
- Preprocessed data using KNNImputer, StandardScaler, and encoding techniques.
- Trained and tuned multiple models (GBM, XGBoost, Lasso, Ridge).
- Used RFECV for feature selection; GBM selected as the best model (based on RMSE & MAPE).
- Interpreted results using SHAP and PDP to uncover cost drivers.
## 📈 Business Recommendations

### For Insurance Companies:
- Use **weight and fat %** to personalize premiums.
- Target **Groups 2 & 3** with **preventive health programs**.
- Offer **wearable-integrated plans** for behavior-based discounts.
- Monitor **weight change** as a leading health risk indicator.

### For Customers:
- Maintain **healthy weight** to reduce premiums.
- Regular **medical checkups** promote early risk detection.
- Avoid rapid **weight fluctuations**.
- Engage in **preventive care** for long-term savings.
## 🧠 Key Learnings

- How to handle high-dimensional, real-world healthcare data
- Applied **KNN imputation**, **feature interaction engineering**, and **model interpretation** techniques
- Built explainable ML models and used clustering for behavior segmentation
- Designed actionable **business strategies** from analytical results
## 📈 Outcome

Recommended risk-based pricing, personalized plans, and wearable-integrated policies for insurers. Identified weight, checkups, and existing coverage as key cost influencers.
