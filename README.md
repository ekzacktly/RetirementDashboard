# Auto-Optimized Retirement Dashboard

## Project Overview
The **Auto-Optimized Retirement Dashboard** is an advanced, highly-interactive retirement simulation engine built with Python and Streamlit. It goes beyond standard compound interest calculators by meticulously simulating the U.S. tax code, IRS withdrawal rules (including RMDs and early withdrawal penalties), dynamic spending phases (the "Retirement Smile"), and optimized withdrawal waterfalls (taxable vs. pre-tax vs. post-tax). 

Detailed information regarding the mathematical models, tax brackets, and withdrawal logic is available within the **User Manual** tab directly on the dashboard. Additionally, a dynamic **The Financial Story** tab provides a personalized, phase-by-phase narrative of your simulation to explain exactly how your money is being managed over time.

## Prerequisites
* **Python:** 3.12.0
* **Package Manager:** `virtualenv` and `pip`

## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   For Windows:
   `python -m virtualenv venv`
   `venv\Scripts\activate`

   For macOS/Linux:
   `python -m virtualenv venv`
   `source venv/bin/activate`

2. **Install Dependencies:**
   Ensure your environment has the required packages installed (Streamlit, Pandas, Numpy-Financial, SciPy, and Plotly).
   `pip install streamlit pandas numpy-financial scipy plotly`

## Running the Dashboard

### 1. Running with Generic Defaults
To run the dashboard using the generic, hardcoded default values (useful for testing or demonstration), simply execute:

`streamlit run dashboard.py`

### 2. Creating and Using a Personal Profile (.json)
To keep your private financial data separate from the codebase, the dashboard supports loading initial states via a JSON profile.

**Step A: Create a Profile JSON File**
Create a new file in the project directory named `my_profile.json`. Here is a basic template structure you can use and adapt to your actual inputs based on the dashboard's parameters:

{
  "current_age": 35,
  "retirement_age": 60,
  "target_lifespan": 95,
  "person1_salary": 75000,
  "person2_salary": 75000,
  "pre_tax_balance_p1": 50000,
  "pre_tax_balance_p2": 50000,
  "roth_balance": 20000,
  "brokerage_balance": 10000,
  "desired_pre_tax_retirement_income": 100000
}

*(Note: Ensure the keys in your JSON file match the variable names expected by the command-line arguments in `dashboard.py`.)*

**Step B: Run the Dashboard with Your Profile**
To inject your personalized settings into the application, use the `--profile` flag. Because Streamlit has its own command-line arguments, you must pass `--` first to separate Streamlit arguments from the script's arguments:

`streamlit run dashboard.py -- --profile my_profile.json`

## Security Note
If you are using version control (like Git), **do not commit your personal profile JSON files**. Be sure to add `*.json` or specifically `my_profile.json` to your `.gitignore` file to ensure your financial data remains private.