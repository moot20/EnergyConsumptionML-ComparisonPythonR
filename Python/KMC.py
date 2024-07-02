import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('datasets/online_shoppers_intention.csv')

data['Month'] = pd.factorize(data['Month'])[0]
data['VisitorType'] = pd.factorize(data['VisitorType'])[0]
data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

features = data.drop(['Revenue'], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

n_clusters = 3
kmeans = KMeans(
    n_clusters=n_clusters,
    init='random',
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)