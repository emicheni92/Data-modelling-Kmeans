
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data name Data
Data = pd.read_csv("Data.csv", encoding="iso-8859-1")

print(Data.head())

print(Data.info())

print(Data.info())

print(Data.columns)

print(Data.describe())

# Check for missing values
print(Data.isnull().sum())

# Calculate percentage of missing values
df_null = round(100 * (Data.isnull().sum() / len(Data)), 2)
print(df_null)

# Drop rows with any missing values
Data = Data.dropna()

# Convert 'CustomerID' to string type
Data['CustomerID'] = Data['CustomerID'].astype(str)

# Verify no more missing values
print(Data.isnull().sum())

# Create 'Amount' column
Data["Amount"] = Data["Quantity"] * Data["UnitPrice"]
print(Data.head())

# Calculate total amount spent by each customer
Data_monetary = Data.groupby("CustomerID")["Amount"].sum().reset_index()
print(Data_monetary.head())

# Calculate total quantity for each product description
Data_monetary = Data.groupby("Description")["Quantity"].sum().reset_index()
print(Data_monetary.head())

# Remove trailing spaces from column names
Data.columns = Data.columns.str.strip()

# Calculate the country with the highest total sales amount
Data_monetary = Data.groupby("Country")["Amount"].sum().idxmax()
print(Data_monetary)

# Calculate the count of invoices for each product description, sorted in descending order
Data_monetary = Data.groupby("Description")["InvoiceNo"].count().sort_values(ascending=False)
print(Data_monetary.head())

# Convert 'InvoiceDate' to datetime
Data["InvoiceDate"] = pd.to_datetime(Data["InvoiceDate"], format="%m/%d/%Y %H:%M")
print(Data.info())

# Calculate date range in the dataset
max_date = max(Data["InvoiceDate"])
print(max_date)
min_date = min(Data["InvoiceDate"])
print(min_date)
days = max_date - min_date
print(days)

# Calculate the cutoff date for the last 30 days
Tsale = max_date - timedelta(days=30)
print(Tsale)

# Calculate the total sales in the last 30 days
total_sales = Data[(Data['InvoiceDate'] >= Tsale) & (Data['InvoiceDate'] <= max_date)]['Amount'].sum()
print("Total Sales of the Last Month:", total_sales)

# KMeans modeling
# Prepare data for KMeans clustering
pro_data = Data.groupby("StockCode").agg({"Quantity": "sum", "UnitPrice": "sum"}).reset_index()

# Feature scaling 
scaler = StandardScaler()
pro_data_scaled = scaler.fit_transform(pro_data[["Quantity", "UnitPrice"]])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
pro_data["Cluster"] = kmeans.fit_predict(pro_data_scaled)

# Evaluate KMeans clustering using silhouette score
Data1= silhouette_score(pro_data_scaled, kmeans.labels_, metric="euclidean")
print("Silhouette Score:", Data1)

# Split the data for a different modeling task predicting the Amount based on Quantity
x_train, x_test, y_train, y_test = train_test_split(Data[["Quantity"]], Data[["UnitPrice"]], test_size=0.3, random_state=42)

# Feature scaling for train and test data
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

print("Train and Test split successful")

# optimal number of clusters using silhouette score
K = range(2, 8)
fits = []
scores = []

for k in K:
    model = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(pro_data_scaled)
    fits.append(model)
    scores.append(silhouette_score(pro_data_scaled, model.labels_, metric="euclidean"))

print(fits)
print(scores)

#plotting lineplot
sns.lineplot(x=K,y=scores)
plt.show()












