import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np 
# Đọc dữ liệu từ tệp CSV vào DataFrame
data = pd.read_csv('Processed_Movies80_processed.csv')

# Kiểm tra dữ liệu
print(data.head())
# đưa dữ liệu dạng số 
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Budget'] = pd.to_numeric(data['Budget'], errors='coerce')
data['DomesticOpening'] = pd.to_numeric(data['DomesticOpening'], errors='coerce')
data['DomesticSales'] = pd.to_numeric(data['DomesticSales'], errors='coerce')
data['InternationalSales'] = pd.to_numeric(data['InternationalSales'], errors='coerce')
data['RunningTime'] = pd.to_numeric(data['RunningTime'], errors='coerce')
# Chuyển đổi DataFrame thành ma trận điểm
ratings = data.pivot_table(index='Title', values=['Year','Budget','DomesticOpening','DomesticSales','InternationalSales','RunningTime']).fillna(0)
# Chuyển đổi thành định dạng ma trận thưa (sparse matrix)
ratings_sparse = csr_matrix(ratings.values)


# Kiểm tra ma trận đánh giá
print(ratings.head())


# Xây dựng mô hình Item-based Collaborative Filtering
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(ratings_sparse)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra

joblib.dump(model, 'item_cf_model.joblib')
