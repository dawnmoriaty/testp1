import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải xuống các resources cần thiết
nltk.download('punkt')
nltk.download('stopwords')

# Tự tạo danh sách stopwords cho tiếng Việt
vietnamese_stopwords = {
    # ... các từ bạn đã có
    'ta', 'mình', 'chúng ta', 'chúng mình', 'của tôi', 'của mình', 'của chúng ta', 'của chúng mình',
    'họ', 'chúng', 'của họ', 'của chúng', 'nó', 'của nó',
    'nay', 'hôm nay', 'hôm qua', 'ngày mai', 'sáng', 'trưa', 'chiều', 'tối', 'lúc', 'khi', 'trước', 'sau', 'rồi', 'sớm',
    'muộn',
    'đây', 'đó', 'này', 'kia', 'đâu', 'nào', 'đến', 'từ', 'qua', 'lên', 'xuống',
    'thôi', 'nào', 'mà', 'chứ', 'nhé', 'à', 'ơi',
    'ôi', 'a', 'hay', 'quá', 'lắm', 'thật',
    'nhiều', 'ít', 'mấy', 'một số', 'các'
}


# Tiền xử lý dữ liệu
def preprocess_text(text, language):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)

    if language == 'english':
        stop_words = set(stopwords.words('english'))
    elif language == 'vietnamese':
        stop_words = vietnamese_stopwords

    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# Đọc dữ liệu đầu vào
data_en = pd.read_csv('data/emails_english.csv')
data_vi = pd.read_csv('data/emails_vietnamese.csv')

# Xử lý dữ liệu
try:
    data_en['text'] = data_en['text'].apply(lambda x: preprocess_text(x, 'english'))
    data_vi['text'] = data_vi['text'].apply(lambda x: preprocess_text(x, 'vietnamese'))
except Exception as e:
    print("Error:", e)


# Hàm để xây dựng và đánh giá mô hình
def build_and_evaluate_model(data, language):
    # Chuyển đổi văn bản thành vector số
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Xây dựng mô hình Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Xây dựng mô hình Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Kết hợp hai mô hình
    ensemble_model = VotingClassifier(estimators=[('nb', nb_model), ('dt', dt_model)], voting='hard')
    ensemble_model.fit(X_train, y_train)

    # Dự đoán và đánh giá mô hình
    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    print(f'Results for {language} model:')
    # Accuracy: Cho biết tỷ lệ tổng thể các email được phân loại đúng.
    print(f'Accuracy: {accuracy}')
    # Precision: Cho biết khi mô hình dự đoán một email là spam, thì xác suất nó thực sự là spam là bao nhiêu.
    print(f'Precision: {precision}')
    #Recall: Cho biết tỷ lệ các email spam thực tế được mô hình phát hiện ra.
    print(f'Recall: {recall}')
    #F1-score: Là một chỉ số tổng hợp, cân bằng giữa precision và recall, thường được sử dụng để đánh giá hiệu suất của mô hình phân loại.
    print(f'F1 Score: {f1}')
    print()

    return ensemble_model, vectorizer


# Xây dựng và đánh giá mô hình cho tiếng Anh
model_en, vectorizer_en = build_and_evaluate_model(data_en, 'English')

# Xây dựng và đánh giá mô hình cho tiếng Việt
model_vi, vectorizer_vi = build_and_evaluate_model(data_vi, 'Vietnamese')


# Hàm để phân loại email mới
def classify_email(email_text, language):
    if language == 'english':
        preprocessed_text = preprocess_text(email_text, 'english')
        vectorized_text = vectorizer_en.transform([preprocessed_text])
        prediction = model_en.predict(vectorized_text)
    elif language == 'vietnamese':
        preprocessed_text = preprocess_text(email_text, 'vietnamese')
        vectorized_text = vectorizer_vi.transform([preprocessed_text])
        prediction = model_vi.predict(vectorized_text)
    else:
        return "Unsupported language"

    return "spam" if prediction[0] == 'spam' else "ham"


# Ví dụ sử dụng
print(classify_email("This is a test email in English", "english"))
print(classify_email("Đây là một email thử nghiệm bằng tiếng Việt", "vietnamese"))
