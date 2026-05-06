# 🍫 Chocolate Sales – Predictive Analytics Dashboard

This project is a fully interactive **Streamlit dashboard** for predictive analytics using the Chocolate Sales dataset.  
It provides:

- Data cleaning & preprocessing  
- **Time series forecasting** (Holt‑Winters exponential smoothing)  
- **Regression modeling** (Linear Regression / Random Forest)  
- Model evaluation (MAE, RMSE, R²)  
- Future trend forecasting and visualisation  

---

## 📁 Dataset

The dashboard uses **`Chocolate Sales.csv`** which must be placed in the same directory as `app.py`.  
Expected columns:

| Column           | Description                          |
|------------------|--------------------------------------|
| Sales Person     | Name of the sales person             |
| Country          | Country of sale                      |
| Product          | Chocolate product name               |
| Date             | Transaction date (format: `dd-Mon-yy`)|
| Amount           | Sale amount (with $ and commas)      |
| Boxes Shipped    | Number of boxes shipped              |

> The file `Chocolate Sales.csv` is included in the project folder.

---

## 🚀 Installation & Setup

1. **Ensure `Chocolate Sales.csv` is in the same folder as `app.py`**  

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate