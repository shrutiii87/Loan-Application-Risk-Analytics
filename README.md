<img src="https://capsule-render.vercel.app/api?type=waving&color=0:020617,10:0F172A,25:1E293B,40:2563EB,55:06B6D4,70:22D3EE,85:A855F7,100:EC4899&height=240&section=header&text=Loan%20Application&fontSize=54&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Turning%20loan%20data%20into%20risk%20insights&descAlignY=60&descSize=20"/>

  
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

#### 🛠️ imported libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import skew, kurtosis
from scipy import stats
```

## 📊 Step : 1 Central Tendency & Dispersion

### (a) Find mean , median , and mode of income .

```python
# Select only subject columns
data = df[["Math", "Physics", "Chemistry", "English", "Computer"]].values

# Student vectors
v1 = data[0]   # S01
v2 = data[1]   # S02

print("Student 1 (S01) Vector:", v1)
print("Student 2 (S02) Vector:", v2)
```
#### 🎯 Insight :-
- Mean Income 📊 → ₹109,639 → Shows the overall average, slightly higher than the median, hinting at some high earners pulling the mean up.

- Median Income 📈 → ₹108,503 → The “middle” income, close to the mean, suggesting the distribution is fairly balanced.

- Mode Income 🔄 → ₹43,670 → The most common salary, much lower than mean/median, showing a cluster of employees earning in this band.

---

#### (b) Calculate range , variance , and standard deviation of Loan_amount . 


