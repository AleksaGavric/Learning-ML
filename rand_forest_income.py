import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_income_data():
    """
    Load the income data from the CSV file.

    Returns:
        DataFrame: The loaded income data.
    """
    income_data = pd.read_csv("income.csv", header=0, delimiter=", ")
    return income_data

def preprocess_income_data(income_data):
    """
    Preprocess the income data by encoding categorical variables and selecting relevant features.

    Args:
        income_data (DataFrame): The income data.

    Returns:
        DataFrame: The preprocessed data.
        DataFrame: The labels.
    """
    labels = income_data[["income"]]
    labels_amount = len(labels)

    # Encode categorical variables
    income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
    income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

    # Select relevant features
    data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex", "country"]]

    return data, labels

def train_and_evaluate_classifier(train_data, test_data, train_labels, test_labels):
    """
    Train and evaluate a Random Forest classifier.

    Args:
        train_data (DataFrame): The training data.
        test_data (DataFrame): The test data.
        train_labels (DataFrame): The training labels.
        test_labels (DataFrame): The test labels.
    """
    classifier = RandomForestClassifier(random_state=1)
    classifier.fit(train_data, train_labels)
    accuracy = classifier.score(test_data, test_labels)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    income_data = load_income_data()
    data, labels = preprocess_income_data(income_data)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)
    train_and_evaluate_classifier(train_data, test_data, train_labels, test_labels)
