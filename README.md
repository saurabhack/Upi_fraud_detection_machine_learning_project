# Indian UPI Customer Segmentation Engine

An end-to-end Machine Learning web application (**Scikit-learn + FastAPI + React**) that segments **2.5 lakh Indian UPI transactions** into 5 actionable customer personas and recommends real-time banking strategies.

**Live APP** :- https://upi-fraud-detection-machine-learning-q1fi.onrender.com/

**App Preview**:- 


<img width="1806" height="967" alt="image" src="https://github.com/user-attachments/assets/759640c8-e37a-4e71-8787-7db3d96022be" />


After clicking on Analyze fraud segment button 

<img width="1806" height="967" alt="image" src="https://github.com/user-attachments/assets/e271dbdd-c374-4c1f-8ec3-cfbd6fee33ef" />

<img width="1806" height="967" alt="image" src="https://github.com/user-attachments/assets/9698c00c-af57-4f8f-bd18-725638a92e5f" />


# Business Problem
Indian UPI apps (PhonePe, Paytm, Google Pay) and banks process billions of monthly transactions, but treating every user the same wastes marketing budgets and strains bank servers.

# This project solves two business goals:
1. **Discover natural spending personas** across 2,50,000 UPI transactions using Unsupervised Machine Learning.
2. **Deploy a real-time prediction API** that classifies any new UPI payment (amount in ₹ + hour of day) and suggests a targeted action—such as promoting **UPI Lite** to micro-spenders or **RuPay Credit Cards / EMI** to high-value payers.


### Key Business Takeaways
* **53% of all transactions (~1.32 lakh)** are small ₹303–₹304 food payments—ideal for shifting to **UPI Lite** to reduce bank server load.
* **Evening rush (6 PM–8 PM)** drives ~22,000 more transactions than morning rush, making it the highest-ROI window for cashback campaigns.
* **Data Debugging Insight:** Initial supervised fraud modeling proved that the dataset's synthetic `fraud_flag` column contained random noise (PR-AUC ~0.0017), leading to a strategic pivot toward high-signal Unsupervised Spending Segmentation.

***Why K-Means over DBSCAN?*** K-Means scales cleanly to 2.5 lakh rows in seconds and supports live .predict() calls in production, whereas DBSCAN lacks a .predict() method and treats rare high-value ₹12,258 payers as noise.

# How to run locally 
1
git clone 
cd upi-customer-segmentation

2
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

3
cd frontend
npm install
npm run dev

***Author*** :- ***Saurabh kamane***
