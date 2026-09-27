# eHealth2025
Repository for ML analysis project @ HES-SO Valais / Wallis
The task was to use different kinds of ML models to try and predict whether or not a patient is susceptible to HPV using data collected from medical images & clinical data.

## Machine learning models used:
- Multilayer Perceptron
- K-nearest Neighbours-
- Multilayer Perceptron (Ensemble)

## Frameworks used
- sci-kit learn
- pandas
- seaborn

The only missing data we identified was in the clinical data dataset:

<img width="505" height="376" alt="Screenshot 2026-09-27 at 23 42 10" src="https://github.com/user-attachments/assets/3f3b38f8-2a48-4610-b51c-10ec28662ccb" />

Values in these columns were all either 0 or 1, so we replaced null values with 0 as a placeholder.

We found that of all factors, tobacco usage was the most important feature in the clinical data:

<img width="559" height="439" alt="image" src="https://github.com/user-attachments/assets/ca8ecd31-20d2-4b71-9172-6d820027574c" />

After testing with our different models, we found that clinical data resulted in the best performance with all models, with MLP being the most accurate:

<img width="691" height="457" alt="image" src="https://github.com/user-attachments/assets/25f81aef-a506-442e-8ee6-b28f2347df15" />

Instructions to setup project:

Disclaimer: depending on python install, you might have to use python instead of python3, or pip instead of pip3

Navigate to directory where you want to clone repo

    git clone <repo>

    # Create virtual environment
    python3 -m venv venv

    # Activate it
    source venv/bin/activate

    # Install required packages
    pip3 install -r requirements.txt
