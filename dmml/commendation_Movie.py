import pandas as pd
from scipy.sparse import csr_matrix
import joblib
from sklearn.model_selection import train_test_split
# Tải mô hình từ tệp
model = joblib.load('item_cf_model.joblib')
data = pd.read_csv('Processed_Movies80_processed.csv')


# Chuyển đổi cột 'DomesticSales' và 'InternationalSales' sang dạng số
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
train_data, test_data = train_test_split(ratings_sparse, test_size=0.2, random_state=42)

# Định nghĩa hàm gợi ý phim cho người dùng
def recommend_movies(user_movie):
    # Tìm các hàng xóm gần nhất trong ma trận đánh giá
    user_movie_index = ratings.index.get_loc(user_movie)
    distances, indices = model.kneighbors(ratings_sparse[user_movie_index], n_neighbors=4)

    # In ra danh sách các phim hàng xóm gần nhất
    print("Các phim gợi ý cho bạn:")
    for i in range(1, len(indices.flatten())):
        recommended_movie = ratings.index[indices.flatten()[i]]
        print(f"{i}. {recommended_movie}")

# Sử dụng mô hình để gợi ý phim cho người dùng
target_movie = 'Avatar'
recommend_movies(target_movie)