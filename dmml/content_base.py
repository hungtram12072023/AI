import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Đọc dữ liệu từ file CSV
df = pd.read_csv('Processed_Movies80.csv')

df['Title'] = df['Title'].fillna('')
df['Features'] = df['Title'] + ' ' + df['Year'].astype(str) + ' ' + df['Budget'].astype(str) + ' ' + df['DomesticOpening'].astype(str) + ' ' + df['DomesticSales'].astype(str) + ' ' + df['InternationalSales'].astype(str) + ' ' + df['RunningTime'].astype(str)

# Xây dựng ma trận đặc trưng
vectorizer = TfidfVectorizer()
features_matrix = vectorizer.fit_transform(df['Features'])

# Tính toán độ tương tự cosine
similarity_matrix = cosine_similarity(features_matrix)

# Lưu ma trận độ tương tự
joblib.dump(similarity_matrix, 'similarity_matrix.joblib')

# Hàm đề xuất phim dựa trên mã phim
#def recommend_movies(movie_id, similarity_matrix, df):

  # Lấy vector đặc trưng của phim đã cho
  #movie_vector = features_matrix[movie_id]

  # Tính toán độ tương tự cosine với tất cả các phim khác
 # cosine_similarities = similarity_matrix[movie_id]

  # Lấy top k phim có độ tương tự cao nhất (bỏ qua phim hiện tại)
  #top_k_movie_indices = cosine_similarities.argsort()[-k-1:-1]  # -k-1 để bỏ qua phim hiện tại

  # Lấy thông tin phim được đề xuất
  #recommended_movies = df.iloc[top_k_movie_indices]

  # Chọn các cột cần thiết
  #recommended_movies = recommended_movies[['Title', 'Year', 'Budget', 'RunningTime']]

  #return recommended_movies

# Ví dụ sử dụng
#movie_id = 1  # Thay đổi mã phim để đề xuất cho phim khác
#k = 10  # Số lượng phim được đề xuất

#recommended_movies = recommend_movies(movie_id, similarity_matrix, df)
#print(f"Phim được đề xuất cho {df.loc[movie_id]['Title']} ({df.loc[movie_id]['Year']}):")
#print(recommended_movies)

