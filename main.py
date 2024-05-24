import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Configuration
NUM_CLUSTERS = 5
NUM_INITIALIZATIONS = 10
FILE_PATH = 'rc_task_2.csv'


# Load the dataset from a CSV file and return the feature columns as a numpy array
def load_dataset(file_path):
    
    df = pd.read_csv(file_path)
    features = df[['feature_1', 'feature_2']].values
    return features

# Initialize the centroids using the k-means++ algorithm
def initialize_centroids(features, k):
    
    # Select the first centroid randomly
    centroids = [features[np.random.randint(0, len(features))]]

    # Select the remaining centroids using the k-means++ algorithm
    for _ in range(1, k):
        distances = np.min(np.array([np.linalg.norm(features - c, axis=1) for c in centroids]), axis=0)
        probabilities = distances ** 2 / np.sum(distances ** 2)
        next_centroid = features[np.random.choice(len(features), p=probabilities)]
        centroids.append(next_centroid)

    return np.array(centroids)

# Perform k-means clustering on the given features using the provided centroids
def k_means(features, centroids):
    
    old_centroids = np.zeros_like(centroids)
    while not np.array_equal(old_centroids, centroids):
        old_centroids = centroids.copy()
        distances = np.array([np.linalg.norm(features - c, axis=1) for c in centroids])
        labels = np.argmin(distances, axis=0)
        centroids = np.array([features[labels == i].mean(axis=0) for i in range(len(centroids))])

    return labels, centroids

# Perform multiple initializations of k-means clustering and return the best clustering result
def k_means_multiple_initializations(features, k, num_initializations):
   
    best_labels = None
    best_centroids = None
    best_score = float('inf')

    for _ in range(num_initializations):
        centroids = initialize_centroids(features, k)
        labels, centroids = k_means(features, centroids)
        score = np.sum((features - centroids[labels]) ** 2)
        if score < best_score:
            best_labels = labels
            best_centroids = centroids
            best_score = score

    return best_labels, best_centroids

# Add cluster labels to the original dataset
def add_cluster_labels(df, labels):
   
    df['team_number'] = labels
    return df
 
# Save the clustered data to a new CSV file
def save_clustered_data(df, file_path):
    
    df.to_csv(file_path, index=False)

# Plot the clusters and centroids
def plot_clusters(features, labels, centroids):
    
    # Plotting the clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # Scatter plot of the data points colored by clusters
    plt.figure(figsize=(8, 6))
    for i in range(NUM_CLUSTERS):
        plt.scatter(features[labels == i, 0], features[labels == i, 1], c=colors[i], label=f'Team {i+1}')

    # Scatter plot of the cluster centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering')
    plt.legend()
    plt.show()


def main():
    # Load the dataset
    features = load_dataset(FILE_PATH)

    # Perform k-means clustering with multiple initializations
    labels, centroids = k_means_multiple_initializations(features, NUM_CLUSTERS, NUM_INITIALIZATIONS)

    # Add cluster labels to the dataset
    df = pd.read_csv(FILE_PATH)
    df = add_cluster_labels(df, labels)

    # Save the clustered data
    save_clustered_data(df, FILE_PATH)

    # Plot the clusters and centroids
    plot_clusters(features, labels, centroids)


if __name__ == '__main__':
    main()