import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

# Tải xuống stopwords
nltk.download('stopwords')


# Tiền xử lý dữ liệu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('vietnamese')]
    return ' '.join(words)

# Giả sử chúng ta có một DataFrame với các cột: 'email_id', 'label', 'text'
data = pd.read_csv('data/emails.csv')

try:
    data['text'] = data['text'].apply(preprocess_text)
except Exception as e:
    print("Error:", e)


# Chuyển đổi văn bản thành vector số
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')