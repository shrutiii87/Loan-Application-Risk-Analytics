<div>
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020617,10:0F172A,25:1E293B,40:2563EB,55:06B6D4,70:22D3EE,85:A855F7,100:EC4899&height=240&section=header&text=Loan%20Application&fontSize=54&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Turning%20loan%20data%20into%20risk%20insights&descAlignY=60&descSize=20"/>
</div>

Analyzing the Loan Application dataset to uncover borrower behavior, income–loan dynamics, and default risk — turning loan data into actionable insights with data science.
✨ Data‑driven insights into loan applications and risk prediction.

---

🎯 Objective :-
The objective of this project is to analyze loan application data to uncover borrower behavior patterns, evaluate income–loan dynamics, and predict default risk using data science techniques. By applying statistical methods and machine learning, the project aims to provide actionable insights into credit risk, helping financial institutions make informed lending decisions and improve risk management strategies.

---

## 🗂️ Project Files

| File                                | Description                                                      |
| ----------------------------------- | ---------------------------------------------------------------- |
| 📓 `loan_application.ipynb`         | Complete data science analysis and Python implementation         |
| 📊 `loan_application_dataset.xlsx`  | Loan Application dataset (borrower income, loan, credit, default)|
| 📄 `Loan_Application_theory.pdf`    | Theory, mathematical derivations, formulas, and interpretations  |
| 🖼️ `Diagrams/`                       | Charts, diagrams, and visualizations of borrower behavior        |
| 📘 `README.md`                      | Project documentation                                            |

---

## 🛠️ Tools Used

<div>
  
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Matplotlib-2C2D72?style=for-the-badge&logo=plotly&logoColor=white"/>

</div>

---

# 🎬 Project Demo

[![Watch Demo](https://img.shields.io/badge/▶️%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1__WPrCETYab-M1IWBLWkbHsIcmElHnY4/view?usp=sharing)

📹 Click the badge above to watch the complete project demonstration.

---

### 🖼️ Image :- 

<img width="960" height="540" alt="Screenshot 2026-07-05 190738" src="https://github.com/user-attachments/assets/3d9e654c-45f4-4ead-9964-af948d8fbda7" />

---

### 📜 Theory part :-

<img width="800" height="420" alt="ezgif-1fde48dcdd9f9d09" src="https://github.com/user-attachments/assets/f0cdb0c5-0254-4d0c-965a-c7d71ae3722a" />

### 📝 Jupyter notebook :- 

<img width="800" height="420" alt="ezgif-8b1e22c8f04559de" src="https://github.com/user-attachments/assets/0010f732-1c7b-4f26-8886-c6a0e196080c" />

---

## 🐍 Part :- B Practical (Python Programming)

---

#### 🛠️ Import libraries :-

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy import stats

print("imported all required libraries !")
```

---

#### 📁 Load the dataset :-

```python
df = pd.read_csv('Loan_applications.csv')

print("File loaded succesfully !")
```

---

#### 🔷 First 5 data :-

```python
df.head()
```

#### 🔷 Last 5 data :-

```python
df.tail()
```

#### 🔷 Describe the data :-

```python
df.describe()
```

#### 🔷 Info of the data :-

```python
df.info()
```

#### 🔷 Shape of the data :-

```python
df.shape
```

#### 🔷 name of the columns :-

```python
df.columns
```

---

## 📊 Step : 1 Central Tendency & Dispersion

#### (a) Find mean , median , and mode of income .

```python
mean_income = df['Income'].mean()
median_income = df['Income'].median()
mode_income = df['Income'].mode()


print(f"Mean Income: {mean_income}")
print(f"Median Income: {median_income}")
print(f"Mode Income: {mode_income}")
```

#### 🎯 Insight :-
- Mean Income 📊 → ₹109,639 → Shows the overall average, slightly higher than the median, hinting at some high earners pulling the mean up.

- Median Income 📈 → ₹108,503 → The “middle” income, close to the mean, suggesting the distribution is fairly balanced.

- Mode Income 🔄 → ₹43,670 → The most common salary, much lower than mean/median, showing a cluster of employees earning in this band.

---

#### (b) Calculate range , variance , and standard deviation of Loan_amount .

```python
loan_range = df['Loan_Amount'].max() - df['Loan_Amount'].min()

loan_variance = df['Loan_Amount'].var()
loan_std = df['Loan_Amount'].std()

print(f"Range: {loan_range}")
print(f"Variance: {loan_variance}")
print(f"Standard Deviation: {loan_std}")
```

#### 🎯 Insight :-
- Range 📏 → 494,647 → Loans vary widely.
- Variance 📐 → 20.4B → Huge spread in loan values.
- Std Dev 📊 → 142,845 → Average loan deviation is very high.
- ✨ Conclusion : Loan amounts are highly inconsistent — big gap between small and - large loans.

---

## 📉 Step : 2 Probability & Events

#### (a) Compute Probability of loan default .

```python
default_count = (df['Default_Status(Yes/No)'] == 'Yes').sum()  #default_count
total_customers = len(df)

prob_default = default_count / total_customers

print("Probability of Loan Default =", prob_default)
```

#### 🎯 Insight :-
- Default Probability 📊 → 0.5602 (≈56%) → More than half of customers are likely to default.
- Risk Signal ⚠️ → High default rate indicates significant credit risk.
- Business Impact 💰 → Lenders must tighten screening or risk heavy losses.
- ✨ Default probability is quite high — strong measures are needed to manage loan risk.

---

#### (b) Creates a contingency table between default_Status and Credit_Score (categorized into ranges) .

```python
df['Credit_Category'] = pd.cut(
    df['Credit_Score'],
    bins=[300, 600, 700, 850],
    labels=['Low', 'Medium', 'High'])

contingency_table = pd.crosstab(        #contingency table
    df['Credit_Category'],
    df['Default_Status(Yes/No)'])

print(contingency_table)
```

#### 🎯 Insight :-
- Low Credit ⚠️ → 2,152 defaults vs 576 non‑defaults → Very high risk group.
- Medium Credit 📉 → 255 defaults vs 658 non‑defaults → Moderate risk, better balance.
- High Credit ✅ → 384 defaults vs 962 non‑defaults → Safest group, majority repay.
- ✨ Default risk drops sharply as credit score improves — strong evidence that credit score is a reliable predictor of repayment behavior.

---

#### (c) Compute conditional probability : P (Default|Credit_score < 600).

```python
low_credit = df[df['Credit_Score'] < 600]

conditional_probability = ((low_credit['Default_Status(Yes/No)'] == 'Yes').sum()/ len(low_credit))  # conditional probability

print("Conditional Probability =", conditional_probability)
```

#### 🎯 Insight :-
- Conditional Probability 📊 → 0.79 (≈79%) → Customers with low credit scores (<600) have a very high chance of defaulting.
- Business Impact 💰 → Strong need for stricter lending policies for this group.
- ✨ Low credit score is a powerful predictor of default — lenders should treat this segment as high‑risk.

---

## 📐 Step :- 3 Distributions & Visulization

#### (a) Plot a histogram of Credit_Score with a Gaussian curve.

<img width="500" height="350" alt="Histogarm of credit score" src="https://github.com/user-attachments/assets/da52cb07-221c-4471-989d-179ac1b3ede4" />

```python
plt.figure(figsize=(8,5))

plt.hist(df['Credit_Score'],bins=20,density=True,alpha=0.6,color='#9932CC', edgecolor='black')

mean = df['Credit_Score'].mean()
std = df['Credit_Score'].std()

x = np.linspace(df['Credit_Score'].min(),df['Credit_Score'].max(),100)

plt.plot(x, norm.pdf(x, mean, std))

plt.title("Histogram of Credit Score")
plt.xlabel("Credit Score")
plt.ylabel("Density")
plt.show()
```

#### 🎯 Insight :-
- Distribution Shape 📈 → Roughly uniform with a slight peak near mid‑range (around 600).
- Spread 📏 → Scores span 300–850, showing wide variation among customers.
- Density Curve 🔵 → Indicates most scores cluster around the average, tapering off at extremes.
- ✨ Credit scores are fairly balanced — not heavily skewed, suggesting a diverse mix of borrower profiles.

---

#### (b) Check a Skewness and Kurtosis of Loan_Amount .

```python
loan_skewness = skew(df['Loan_Amount'])
loan_kurtosis = kurtosis(df['Loan_Amount'])

print("Skewness =", loan_skewness)
print("Kurtosis =", loan_kurtosis)
```

#### 🎯 Insight :-
- Skewness ⚖️ → 0.008 → Almost zero, meaning the loan distribution is nearly symmetric.
- Kurtosis 📉 → −1.20 → Indicates a flat (platykurtic) shape — fewer extreme loan values than a normal curve.
- ✨ Loan amounts are evenly spread with minimal skew and light tails — suggesting stable, moderate variation among borrowers.

---

#### (c) Draw a Q-Q plot for income .

<img width="600" height="455" alt="Q-Q plot of income" src="https://github.com/user-attachments/assets/d2bf4a7f-2ffb-4988-a062-d3f318edeace" />

```python
stats.probplot(df['Income'], dist="norm", plot=plt)

plt.title("Q-Q Plot of Income")
plt.show()
```

#### 🎯 Insight :-
- Normality Check 📈 → Points deviate from the red line at both ends, showing the data isn’t perfectly normal.
- Tail Behavior 💰 → Curved tails indicate right skew — a few high‑income outliers.
- Interpretation 🔍 → Most incomes cluster near the center, but extremes pull the curve upward.
- ✨ Income distribution is moderately skewed — transformation (like Box‑Cox or log) could help normalize it for better statistical modeling.

---

## 🧾 Step :- 4 Linear Algebra Application

#### 🔷 Take the first 5 customer's [Income_Loan_Amount] as vectors .

```python
vectors = df[['Income', 'Loan_Amount']].head(5)
print(vectors)
```

#### (a) Perform dot product between two customer vectors .

```python
v1 = vectors.iloc[0].values
v2 = vectors.iloc[1].values

print(v1)
print(v2)

dot_product = np.dot(v1, v2)

print("Dot Product =", dot_product)
```

#### 🎯 Insight :-
- Skewness 📐 → Positive → Vectors point in the same direction.
- Dot Product 📊 → 8.82B → Very large, showing strong similarity.
- ✨ Both borrowers have aligned income–loan profiles, reinforcing correlation between earnings and loan size.

---

#### (b) Find Norm 2 of a customer's financial vector .

```python
norm = np.linalg.norm(v1)

print("L2 Norm =", norm)
```

#### 🎯 Insight :-
- L2 Norm 📏 → 52,413 → Represents the overall “magnitude” of the borrower’s vector (Income + Loan).
- Interpretation 🔍 → A large norm shows this borrower has a strong financial profile in terms of both income and loan size.
- ✨Norm acts like the “length” of the financial vector — bigger values mean higher combined scale of income and loan.

---

#### (c) Calculate the angle between two customer's vectors .

```python
cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

angle = np.degrees(np.arccos(cos_theta))

print("Angle =", angle, "degrees")
```

#### 🎯 Insight :-
- Angle 📐 → 62.36° → The vectors are not perfectly aligned, showing noticeable difference in borrower profiles.
- Cosine Similarity 🔗 → Moderate similarity — they share some proportionality but diverge significantly.
- ✨ Insight: These two borrowers have related but distinct income–loan patterns, not pointing in the same financial direction.

---

# 🚀 How to Run

Install the required libraries:

```bash
pip install pandas numpy scipy matplotlib
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebook:

```text
loan_application.ipynb
```

---

## 👩‍💻 Shruti Bhawsar

📍 Ahmedabad | 

[![GitHub](https://img.shields.io/badge/GitHub-shrutiii87-181717?style=flat&logo=github)](https://github.com/shrutiii87)

---

⭐ **If You Like This Project**

Give this repository a ⭐ and feel free to fork or contribute!

📊 *Clean Models · Sharp DAX · Confident Insights*
