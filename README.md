# Nifty/Dow Jones Direction Prediction using Machine Learning Algorithms

## Project Overview

This project aims to predict the direction of the Nifty 50 index opening using machine learning algorithms based on previous day's global market returns. The analysis incorporates data from six major global indices (Nifty 50, Dow Jones, Nasdaq, Hang Seng, Nikkei 225, and DAX) along with the VIX volatility index.


## Project Structure

```
PROJECT/
├── README.md                           # This file
├── pull-financial-full.py             # Script to download market data from Yahoo Finance
├── phase1.ipynb                       # Phase 1: Data Collection & Preprocessing
├── phase2.ipynb                       # Phase 2: Exploratory Data Analysis
├── phase3.ipynb                       # Phase 3: Logistic Regression Modeling
├── phase4.ipynb                       # Phase 4: Advanced ML Models
├── phase5.ipynb                       # Phase 5: Text Mining & Sentiment Analysis
├── four-markets.csv                   # Raw market data (generated)
├── markets_with_returns.csv           # Processed data with returns (generated)
├── presentation_guidelines.py         # Script to generate PowerPoint presentation
└── basic-navigation/
    ├── app.py                         # Shiny web application
    ├── shared.py                      # Shared data and utilities
    ├── phase1.py                      # Phase 1 visualizations
    ├── phase2.py                      # Phase 2 visualizations
    ├── phase3.py                      # Phase 3 visualizations
    ├── phase4.py                      # Phase 4 visualizations
    ├── phase5.py                      # Phase 5 visualizations
    └── styles.css                     # CSS styling for Shiny app
```

## Data Collection

### Prerequisites

Install required packages:

```bash
pip install yfinance pandas numpy matplotlib seaborn statsmodels scikit-learn nltk textblob wordcloud python-pptx
```

### Download Market Data

Run the data collection script to download historical market data from Yahoo Finance:

```bash
python pull-financial-full.py
```

This script:
- Downloads daily market data from January 2019 to April 2025
- Retrieves Open and Close prices for 7 indices (Nifty 50, Dow Jones, Nasdaq, Hang Seng, Nikkei 225, DAX, VIX)
- Handles missing values using linear interpolation
- Saves the consolidated data to `four-markets.csv`

**Key Features:**
- Full date range including weekends and holidays
- Interpolated missing values for continuous data
- Separate columns for each market's Open and Close prices

## Project Phases

### Phase 1: Data Collection and Preprocessing
**Jupyter Notebook:** `phase1.ipynb`

- Load and merge market data from multiple sources
- Calculate daily returns for all indices
- Extract temporal features (Year, Quarter, Month)
- Handle missing values and outliers
- Perform normality tests (Shapiro-Wilk, Lilliefors)
- Generate QQ plots for distribution analysis

**Key Outputs:**
- `markets_with_returns.csv` - Processed dataset with calculated returns
- Statistical summaries and normality test results

### Phase 2: Exploratory Data Analysis
**Jupyter Notebook:** `phase2.ipynb`

- Box plots of market returns by year
- Statistical tables (mean, std deviation) by year
- Median return bar plots
- Correlation matrix analysis
- Heatmaps for quarterly trends
- COVID-19 impact analysis
- Nifty 50 directional movement analysis

**Key Insights:**
- Cross-market correlations
- Seasonal patterns and trends
- Market behavior during major events (COVID-19)

### Phase 3: Predictive Modeling (Logistic Regression)
**Jupyter Notebook:** `phase3.ipynb`

- Define binary target variable: Nifty_Dir_Open (1=Up, 0=Down)
- Train/test split (80/20)
- Build logistic regression model
- VIF analysis for multicollinearity
- Feature selection based on statistical significance
- ROC curve analysis (train and test)
- Confusion matrix evaluation
- Optimal threshold determination (0.4753)

**Key Findings:**
- Low AUC (~0.53-0.59) indicates limited predictive power
- Model performance slightly better than random guessing
- Significant variables: HangSeng, Nikkei, DAX, VIX returns

### Phase 4: Advanced Machine Learning Models
**Jupyter Notebook:** `phase4.ipynb`

- **Gaussian Naive Bayes:** Probabilistic classifier with independence assumption
- **Decision Tree:** Non-parametric model with interpretable rules
- **Random Forest:** Ensemble method reducing overfitting
- Model comparison and performance metrics
- Feature importance analysis

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC curves and AUC scores
- Confusion matrices for all models

### Phase 5: Text Mining and Sentiment Analysis
**Jupyter Notebook:** `phase5.ipynb`

- Twitter/X data collection (Nifty 50 related posts)
- Text preprocessing pipeline:
  - Lowercase conversion
  - Remove mentions, hashtags, URLs
  - Remove punctuation and digits
  - Stopword removal
  - Domain-specific word filtering
- Word cloud generation
- VADER sentiment analysis
- Sentiment classification (Positive, Negative, Neutral)

**NLP Techniques:**
- Tokenization and lemmatization
- TF-IDF analysis
- Sentiment scoring integration with market data

## Interactive Shiny Application

### Running the Shiny App

Navigate to the `basic-navigation` directory and run:

```bash
cd basic-navigation
shiny run app.py
```

The app provides an interactive dashboard with:

- **Phase 1:** Data overview, QQ plots, normality tests
- **Phase 2:** Market performance visualizations, correlation analysis, COVID impact
- **Phase 3:** Logistic regression results, ROC curves, confusion matrices
- **Phase 4:** Comparison of ML models with performance metrics
- **Phase 5:** Text mining results, word clouds, sentiment analysis

**Features:**
- Interactive visualizations
- Source code display for reproducibility
- Organized navigation by project phase
- Responsive design with custom CSS styling

## Results Summary

### Model Performance Comparison

| Model | Train AUC | Test AUC | Train Accuracy | Test Accuracy |
|-------|-----------|----------|----------------|---------------|
| Logistic Regression | 0.59 | 0.53 | 56% | 57% |
| Naive Bayes | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |

### Key Insights

1. **Market Interconnectivity:** Strong correlations exist between global markets, particularly Asian markets (Hang Seng, Nikkei) with Nifty 50
2. **Predictive Limitations:** Financial markets show inherent randomness; achieving >60% accuracy is challenging
3. **Feature Importance:** VIX (volatility) and Asian market returns are most significant predictors
4. **COVID-19 Impact:** Significant market volatility observed in Q1-Q2 2020 with subsequent recovery

### Recommendations

- **For Traders:** Use model predictions as one of multiple indicators, not sole decision criterion
- **Risk Management:** Focus on VIX levels and Asian market trends for Nifty direction insights
- **Future Work:**
  - Incorporate additional features (volume, technical indicators)
  - Experiment with deep learning models (LSTM, GRU)
  - Integrate real-time sentiment analysis
  - Develop pairs trading strategies

## Generating Project Presentation

Create a PowerPoint presentation following project guidelines:

```bash
python presentation_guidelines.py
```

This generates `Project_Presentation.pptx` with:
- Title slide with project details
- Background and motivation
- Data sources and methodology
- Analysis plan and phases
- Challenges encountered
- Project timeline

## Technical Requirements

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn, nltk, textblob, wordcloud, yfinance, python-pptx, shiny

## License

This project is for academic purposes as part of the Data Science & Business Analytics course.

## Contact

For questions or collaboration, please contact the project team members.