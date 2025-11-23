# PowerPoint Presentation Generator

This script creates a comprehensive PowerPoint presentation from your Market Analysis Dashboard.

## Installation

First, install python-pptx:

```bash
pip install python-pptx
```

Or install all requirements:

```bash
pip install -r requirements_presentation.txt
```

## Usage

Run the script from the PROJECT directory:

```bash
cd /home/pablosan/Documents/Ireland/study/DSI/PROJECT
python presentation.py
```

## What it Creates

The presentation includes:

### Phase 1: Data Collection and Preprocessing
- Dataset overview
- QQ plots for normality check

### Phase 2: Exploratory Data Analysis
- Box-whisker plots by year
- Correlation matrix
- COVID-19 impact analysis
- Nifty direction vs global indices

### Phase 3: Logistic Regression
- Model approach and methodology
- ROC curves (train & test)
- Confusion matrices (train & test)

### Phase 4: Machine Learning Models
- Naive Bayes analysis
- Decision Tree analysis
- Random Forest analysis
- Model comparison table

### Phase 5: Text Mining & Sentiment Analysis
- Word cloud visualization
- Sentiment distribution charts
- VADER analysis results

### Conclusions
- Key findings
- Recommendations

## Output

The script generates:
- `Market_Analysis_Presentation.pptx` - The final presentation
- `temp_plots/` directory - Temporary plot images (can be deleted after)

## Features

- ✅ Automatically generates all plots from your analysis
- ✅ Professional slide layouts with consistent formatting
- ✅ Color-coded sections
- ✅ Comprehensive coverage of all 5 phases
- ✅ Ready for presentation or export to PDF

## Customization

You can modify the script to:
- Change colors (see DARK_BLUE, LIGHT_BLUE constants)
- Adjust font sizes (see TITLE_FONT_SIZE, etc.)
- Add or remove slides
- Change slide layouts
- Add custom content

## Troubleshooting

If you get import errors, make sure you're in the correct directory and all dependencies are installed.

If plots don't appear, check that all phase modules can generate their plots independently.
