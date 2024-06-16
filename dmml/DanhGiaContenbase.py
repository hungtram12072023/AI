import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
# Đọc dữ liệu phim từ file CSV
df = pd.read_csv('Processed_Movies80.csv')
# Tải ma trận độ tương similarity
similarity_matrix = joblib.load('similarity_matrix.joblib')
def evaluate_model(similarity_matrix, df):

  # Lấy số lượng phim
  n_movies = similarity_matrix.shape[0]

  # Tạo ma trận đánh giá
  # - Đoạn code này tạo ra một ma trận đánh giá (rating_matrix) ngẫu nhiên với kích thước (n_movies, n_movies)
  # - Mỗi phần tử trong ma trận này là một đánh giá ngẫu nhiên cho cặp phim tương ứng (i, j)
  rating_matrix = np.random.randint(1, 6, size=(n_movies, n_movies))

  # Tính toán dự đoán
  # - Đoạn code này sử dụng ma trận độ tương similarity để dự đoán đánh giá cho mỗi cặp phim
  # - Ví dụ: dự đoán đánh giá của người dùng cho phim j là trung bình cộng của đánh giá cho phim j từ các phim tương tự với nó
  predicted_ratings = similarity_matrix.dot(rating_matrix)

  # Tính toán MSE và MAE
  mse = mean_squared_error(rating_matrix.flatten(), predicted_ratings.flatten())
  mae = mean_absolute_error(rating_matrix.flatten(), predicted_ratings.flatten())

  # Hiển thị kết quả đánh giá
  print("Kết quả đánh giá mô hình:")
  print(f"MSE: {mse:.4f}")
  print(f"MAE: {mae:.4f}")

# Ví dụ sử dụng
evaluate_model(similarity_matrix, df)
