import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import numpy as np

# Tải mô hình từ tệp
model = joblib.load('item_cf_model.joblib')
data = pd.read_csv('Processed_Movies80.csv')

# Chuyển đổi DataFrame thành ma trận điểm
ratings = data.pivot_table(index='Title', values=['Year','Budget','DomesticOpening','DomesticSales','InternationalSales','RunningTime']).fillna(0)

# Chuyển đổi thành định dạng ma trận thưa (sparse matrix)
ratings_sparse = csr_matrix(ratings.values)

def predict_ratings(model, ratings_sparse):
    # Tìm các láng giềng gần nhất cho mỗi mục trong ma trận thưa
    distances, indices = model.kneighbors(ratings_sparse)

    # Khởi tạo một mảng 2D có kích thước giống với ma trận xếp hạng để lưu trữ các dự đoán
    predicted_ratings = []
    for i in range(ratings_sparse.shape[0]):
        neighborhood_ratings = ratings_sparse[indices[i]].toarray()  # Lấy xếp hạng của các láng giềng
        neighborhood_distances = distances[i].reshape(-1, 1)  # Định hình lại ma trận khoảng cách thành 2D

        # Áp dụng trọng số dựa trên khoảng cách và xếp hạng của láng giềng
        weighted_sum = (neighborhood_ratings / (neighborhood_distances + 1e-9)).sum(axis=0)
        sum_of_weights = (1 / (neighborhood_distances + 1e-9)).sum(axis=0)

        # Tránh chia cho 0 và chuẩn hóa để có giá trị từ 0 đến 1
        predicted_rating = np.divide(weighted_sum, sum_of_weights, out=np.zeros_like(weighted_sum), where=sum_of_weights!=0)
        predicted_ratings.append(predicted_rating)

    return np.array(predicted_ratings)
def evaluate_model(model, ratings_sparse):
    # Dự đoán xếp hạng
    predicted_ratings = predict_ratings(model, ratings_sparse)
    
    # Chuyển đổi ma trận thưa thành ma trận mật độ
    actual_ratings = ratings_sparse.toarray()

    # Đánh giá mô hình
    mse = mean_squared_error(actual_ratings.flatten(), np.concatenate(predicted_ratings))
    mae = mean_absolute_error(actual_ratings.flatten(), np.concatenate(predicted_ratings))

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)

evaluate_model(model, ratings_sparse)
