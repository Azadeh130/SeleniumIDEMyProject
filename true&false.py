import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# دانلود استاپ وردز
nltk.download('stopwords')

# خواندن فایل اکسل
data = pd.read_excel('texts.xlsx')

# فرض بر این است که ستون 'text' شامل متن‌ها است و ستون 'label' دارای 0 (برای متن‌های مناسب) و 1 (برای متن‌های نامناسب) است
texts = data['text']
labels = data['label']

# پیش پردازش اولیه: حذف استاپ وردز
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

texts = texts.apply(preprocess_text)

# استفاده از Tokenizer برای تبدیل متن‌ها به توکن‌ها
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# پدینگ متن‌ها به طول ثابت
max_len = 100
X = pad_sequences(sequences, maxlen=max_len)

# تقسیم داده‌ها به مجموعه آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# ساخت شبکه عصبی ساده
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(Flatten())  # تخت کردن داده‌ها برای ورودی به لایه Dense
model.add(Dense(128, activation='relu'))  # لایه مخفی
model.add(Dense(1, activation='sigmoid'))  # لایه خروجی با تابع فعال‌سازی سیگموید برای طبقه‌بندی دودویی

# کامپایل کردن مدل
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# آموزش مدل
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# پیش‌بینی بر روی داده‌های تست
y_pred = model.predict(X_test)

# تبدیل خروجی‌های مدل به 0 و 1
y_pred = (y_pred > 0.5).astype(int)

# ارزیابی مدل
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# بررسی یک فایل اکسل جدید
new_data = pd.read_excel('new_texts.xlsx')
new_texts = new_data['text'].apply(preprocess_text)
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_X = pad_sequences(new_sequences, maxlen=max_len)

# پیش‌بینی
new_predictions = model.predict(new_X)
new_data['prediction'] = (new_predictions > 0.5).astype(int)

# ذخیره نتایج در یک فایل اکسل جدید
new_data.to_excel('classified_texts.xlsx', index=False)

import os
print(os.getcwd())
