import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
flags = pd.read_csv("flags.csv", header=0)

# Extract the target labels from the DataFrame
labels = flags[["Landmass"]]

# Extract the feature data from the DataFrame
data = flags[
    [
        "Red",
        "Green",
        "Blue",
        "Gold",
        "White",
        "Black",
        "Orange",
        "Circles",
        "Crosses",
        "Saltires",
        "Quarters",
        "Sunstars",
        "Crescent",
        "Triangle",
    ]
]

# Initialize an empty list to store the scores
scores = []

# Iterate over different max_depth values for the DecisionTreeClassifier
for max_depth in range(1, 21):
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=1)

    # Create a DecisionTreeClassifier with the specified max_depth
    tree = DecisionTreeClassifier(random_state=1, max_depth=max_depth)

    # Fit the model to the training data
    tree.fit(train_data, train_labels)

    # Calculate and store the accuracy score on the testing data
    scores.append(tree.score(test_data, test_labels))

# Plot the accuracy scores against the max_depth values
plt.plot(range(1, 21), scores)
plt.xlabel("max_depth")
plt.ylabel("Accuracy Score")
plt.title("Decision Tree Classifier Performance")
plt.show()
