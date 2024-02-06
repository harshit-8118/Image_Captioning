"""
Image Caption BOT
Original file is located at
    https://colab.research.google.com/drive/1Qmd57M2pL_jawpbF2jgc89l3isPgegri
"""

import pickle
from numpy import expand_dims
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences

model = ResNet50(weights='imagenet', input_shape=(224,224,3))
model_resnet = Model(model.input, model.layers[-2].output)

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = expand_dims(img, axis=0)

    img = preprocess_input(img)
    return img

def encode_images(img_path):
    img = preprocess_img(img_path)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape((1, 2048))
    return feature_vector

w2i = 'word_index/word_2_idx.pkl'
i2w = 'word_index/idx_2_word.pkl'
model = load_model('models/models_19.h5')

word_2_idx, idx_2_word = None, None
max_len = 35

with open(w2i, 'rb') as f:
    word_2_idx = pickle.load(f)

with open(i2w, 'rb') as f:
    idx_2_word = pickle.load(f)


def predict_caption(photo, max_len, word_2_idx, idx_2_word):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_2_idx[w] for w in in_text.split() if w in word_2_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')[0]
        sequence = expand_dims(sequence, axis=0)
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax(axis=-1)[0]
        word = idx_2_word[ypred]
        in_text += (' ' + word)
        if word == "endseq":
            break
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


def predict_this_image(image):
    photo = encode_images(image)
    caption = predict_caption(photo, max_len, word_2_idx, idx_2_word)
    return caption

