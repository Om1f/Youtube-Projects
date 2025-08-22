import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_datasets as tfds
import os
import pickle


model_path = 'ag_news_model.keras'
tokenizer_path = 'tokenizerforornek2.pkl'

if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    print("Hata: Model veya tokenizer dosyası bulunamadı.")

    exit()



model = keras.models.load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
print("Model ve tokenizer yüklendi")


info = tfds.load('ag_news_subset', with_info=True)[1]
labels_info = info.features['label'].names
max_len = 30

print(f"Kategoriler: {labels_info}")


print("\nBir haber metni girin ('exit' yazarak çıkabilirsiniz):")
while True:
    user_input = input("> ")

    if user_input.lower() == 'exit':
        print("Programdan çıkılıyor...")
        break


    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded = keras.preprocessing.sequence.pad_sequences(user_sequence, padding='post', maxlen=max_len)


    prediction = model.predict(user_padded)
    predicted_label_index = np.argmax(prediction)
    predicted_category = labels_info[predicted_label_index]


    print(
        f"Girdiğiniz metin, en yüksek olasılıkla '{predicted_category}' kategorisine ait. (Olasılık: {prediction[0][predicted_label_index]:.2f})")