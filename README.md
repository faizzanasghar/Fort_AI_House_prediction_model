# Fort_AI_House_prediction_model
# FORT REAL ESTATE AGENT
## Elite Luxury Property Valuation System

**FORT REAL ESTATE AGENT** is a state-of-the-art AI-powered application designed for the valuation and analysis of luxury real estate properties. Built with a premium "Midnight Gold" aesthetic, it combines advanced machine learning with an immersive user experience.

---

##  Key Features

### 1. Ultra-Luxury User Interface
*   **Premium Design**: Features a "Jewel Tone" theme (Deep Navy #0F172A, Gold #D4AF37) with glassmorphism effects.
*   **Navigation Drawer**: A sleek, vertical sidebar drawer for property configuration and page navigation.
*   **Responsive**: Fully mobile-responsive layout with custom animated hamburger menu.

### 2.  AI-Powered Valuation
*   **Advanced Model**: Utilizes a Random Forest Regressor trained on real estate data (King County House Data).
*   **Real-Time Estimates**: Generates instant property value estimates based on user inputs.
*   **Confidence Intervals**: Provides price ranges and confidence scores to aid decision-making.

### 3.  Interactive Market Analytics
*   **Dynamic Charts**: Powered by Plotly for interactive data visualization.
*   **Market Comparison**: Scatter plots comparing your property against the market.
*   **Location Analysis**: Detailed price distribution by location tier (Rural, Suburban, Downtown).

### 4.  Elite Navigation
*   **Price Estimation**: The core valuation dashboard.
*   **Market Analytics**: Deep dive into market trends and statistics.
*   **AI Intelligence**: Insights into model performance and feature importance.

---

##  Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package manager)

### Dependencies
Install the required libraries using pip:

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib
```

### Files
*   `app1.py`: Main application source code.
*   `logo.jpg`: Company branding logo.
*   `kc_House_Data.csv`: Dataset for model training (if using live training).
*   `house_price_model.pkl`: Pre-trained model file (optional, app will retrain if missing).

---

##  How to Run

1.  Navigate to the project directory in your terminal:
    ```bash
    cd path/to/project
    ```

2.  Launch the application using Streamlit:
    ```bash
    streamlit run app1.py
    ```

3.  The app will open automatically in your default web browser (usually at `http://localhost:8501`).

---

##  User Guide

### 1. Navigation
*   Click the **Gold Hamburger Icon** (top-left) to open the **Navigation Drawer**.
*   Select a module: **PRICE ESTIMATION**, **MARKET ANALYTICS**, or **AI INTELLIGENCE**.

### 2. Configuring a Property
*   Open the **Navigation Drawer**.
*   Scroll to ** PROPERTY CONFIG**.
*   Expand sections to adjust estimating factors:
    *   ** Dimensions**: Set Squre footage, bedrooms, bathrooms, and floors.
    *   ** Luxury Factors**: Adjust construction grade, view quality, and condition.
    *   ** Location Tier**: Select the property zone (Rural, Suburban, Downtown).

### 3. Getting a Valuation
*   Navigate to the **PRICE ESTIMATION** page.
*   Click the **CALCULATE PROPERTY VALUE** button.
*   View the estimated price, confidence interval, and executive summary.

---

## Security & Privacy
*   **Local Processing**: All data inputs are processed locally within your session.
*   **Encryption**: The system architecture supports AES-256 standards for data handling.
*   **Confidentiality**: Designed for elite client privacy.

---

**Developed by Muhammad Faizan Asghar**
*© 2025 Fort Real Estate Intelligence*
