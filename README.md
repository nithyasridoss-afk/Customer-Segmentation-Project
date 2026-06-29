# Customer-Segmentation-Project
Segment customers based on behavior and demographicsKey Features:Perform clustering using tools like Python (scikit-learn) or Power BIAnalyze purchase patterns and customer preferencesVisualize segments and key characteristicsExpected Outcome:Gain experience in customer analytics, segmentation, and targeted insights
# This project aims to segment customers into different groups based on their demographic information and purchasing behavior. By applying machine learning clustering algorithms, businesses can identify customer categories and develop targeted marketing strategies.
# Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
# Features
Data collection and preprocessing
Customer behavior analysis
Data visualization
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(8,6))
plt.scatter(data['Annual Income (k$)'],
            data['Spending Score (1-100)'],
            c=data['Cluster'],
            cmap='viridis')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

# Expected Output
Customers grouped into 5 segments.
Visual representation of customer clusters.
Identification of high-value, average-value, and low-value customers.
Better understanding of customer preferences and buying patterns.
# Internship Outcome
Experience with machine learning algorithms.
Knowledge of customer analytics.
Data visualization skills.
Real-world business problem-solving experience.
Portfolio-ready data science project.


