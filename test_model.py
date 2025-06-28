import pytest
from joblib import load
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    print("Test Setup")

@pytest.fixture(scope="module")
def trained_model():
    try:
        return load('model_dt.joblib')
    except:
        return None

@pytest.fixture(scope="module")
def test_data():
    try:
        data = pd.read_csv('./iris.csv')

        train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
        X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
        y_test = test.species
        return X_test, y_test
    except:
        print("test_data error")
        return None, None

def test_model_loading(trained_model):
    """Test if the model loads successfully."""
    assert trained_model is not None
    print("Model Loaded")

def test_data_validation(test_data):
    X_test, y_test = test_data
    assert len(X_test)>0
    assert X_test.shape[0] > 0 # Check if rows exist
    print("Test Data OK")

def evaluate_model(trained_model, test_data) -> dict:
    X_test, y_test = test_data
    y_pred = trained_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        # For multi-class classification, specify 'average' for precision, recall, f1
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro')
    }
    print("Model evaluation complete.")
    print(str(metrics))
    return metrics

def test_evaluate_model(trained_model, test_data) -> dict:
    metrics = evaluate_model(trained_model, test_data)
    expected_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    assert all(metric in metrics for metric in expected_metrics)
    assert isinstance(metrics['accuracy'], float)
    print("Model Eval OK")
