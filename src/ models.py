# Model training functions for the Rain Prediction project.
# Contains two supervised learning algorithms
# 1. Decision Tree Classifier
# 2. Random Forest Classifier
#
# Both models are trained on the training set only.
#
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import DT_PARAMS, RF_PARAMS


def train_decision_tree(X_train_scaled, y_train) -> DecisionTreeClassifier:
    """
    Train a Decision Tree Classifier on the training data.
    Parameters
    X_train_scaled : np.ndarray Scaled training features
    y_train : pd.Series Training labels
    Returns
    DecisionTreeClassifier Trained model
    """
    print('Training Decision Tree...')

    # Initialise with hyperparameters from config
    dt_model = DecisionTreeClassifier(**DT_PARAMS)

    # Train the model — it learns the pattern:
    # "Given today's weather values → predict if it rains tomorrow"
    dt_model.fit(X_train_scaled, y_train)

    # Print tree statistics
    print(f'  Tree Depth      : {dt_model.get_depth()}')
    print(f'  Number of Leaves: {dt_model.get_n_leaves()}')
    print('Decision Tree training complete.')

    return dt_model

def train_random_forest(X_train_scaled, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the training data.

    Parameters
    ----------
    X_train_scaled : np.ndarray  Scaled training features
    y_train        : pd.Series   Training labels

    Returns
    -------
    RandomForestClassifier  Trained model
    """
    print('Training Random Forest...')

    # Initialise with hyperparameters from config
    rf_model = RandomForestClassifier(**RF_PARAMS)

    # Train the model — 100 trees are built in parallel on bootstrap samples
    rf_model.fit(X_train_scaled, y_train)

    print(f'  Number of Trees : {rf_model.n_estimators}')
    print(f'  Features/Split  : {rf_model.max_features}')
    print('Random Forest training complete.')

    return rf_model