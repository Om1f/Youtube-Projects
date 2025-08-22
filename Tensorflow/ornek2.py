import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
import tensorflow_datasets as tfds
#################################
import pickle
model_path = 'ag_news_model.keras'
tokenizer_path = 'tokenizerforornek2.pkl'
#############################
print("1. Veri seti indiriliyor ve hazırlanıyor...")
(train_ds, test_ds), info = tfds.load(
    'ag_news_subset',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

train_ds = train_ds.take(10000)
test_ds = test_ds.take(2000)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_texts = []
train_labels = []
for text, label in train_ds:
    train_texts.append(text.numpy().decode('utf-8'))
    train_labels.append(label.numpy())

test_texts = []
test_labels = []
for text, label in test_ds:
    test_texts.append(text.numpy().decode('utf-8'))
    test_labels.append(label.numpy())

tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<unk>")
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

max_len = 30
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post', maxlen=max_len)
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, padding='post', maxlen=max_len)

model = Sequential([
    layers.Embedding(len(word_index) + 1, 16),
    layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Model eğitimi başlıyor...")
model.fit(train_padded, np.array(train_labels), epochs=10, validation_data=(test_padded, np.array(test_labels)))
print("\nModel eğitimi tamamlandı!")

model.save(model_path)
print(f"\nModel '{model_path}' dosyasına kaydedildi.")
#--------------------------------------------
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Tokenizer '{tokenizer_path}' dosyasına kaydedildi.")
#--------------------------------------------------------
