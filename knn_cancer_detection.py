from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


def evaluate_knn_classifier():
    """
    Evaluate the K-Nearest Neighbors classifier for various values of k and plot the validation accuracy.
    """
    breast_cancer_data = load_breast_cancer()

    # Split the data into training and validation sets
    (
        training_data,
        validation_data,
        training_labels,
        validation_labels,
    ) = train_test_split(
        breast_cancer_data.data,
        breast_cancer_data.target,
        test_size=0.2,
        random_state=100,
    )

    best_k = [0, 0]  # [k, score]
    accuracies = []

    for k in range(1, 101):
        # Create and train the K-Nearest Neighbors classifier
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(training_data, training_labels)

        # Evaluate the classifier on the validation set
        score = classifier.score(validation_data, validation_labels)

        if score > best_k[1]:
            best_k[0] = k
            best_k[1] = score

        accuracies.append(score)

    print("Best k:", best_k)

    # Plotting the validation accuracy against different values of k
    k_list = range(1, 101)

    plt.scatter(k_list, accuracies)
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("K-Nearest Neighbors Classifier Accuracy")
    plt.show()


if __name__ == "__main__":
    evaluate_knn_classifier()
