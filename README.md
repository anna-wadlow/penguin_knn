# Palmer Penguins Species Classifier

This Streamlit application demonstrates the K-Nearest Neighbors (kNN)
classification algorithm using the Palmer penguins dataset available: https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv.
Given a penguin's measurements, island, sex, and year observed, the app predicts its species (Adelie, Gentoo, or Chinstrap).

## Links
- **Live App:** https://penguinknn.streamlit.app/
- **GitHub repository:** https://github.com/anna-wadlow/penguin_knn

## Features

- Choose a random state for the train/test split
- Choose 7 penguin characteristics and predict species for a new observation
- See the value of K selected by 5-fold cross-validation, in addition to CV and test accuracy
- Display predicted probabilities for each class
- Display explanation of prediction

## Files
- 'app.py': The Streamlit application
- 'requirements.txt': Python packages required to run the app
- 'analysis.ipynb': The analysis notebook

## Run locally

Install the required packages:

```bash
pip install -r requirements.txt
```
Start the app:
```
streamlit run app.py
```

The app opens in your browser at http://localhost:8501/