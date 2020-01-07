import pandas as pd
import numpy as np
import pickle
from keras.layers import Input, CuDNNLSTM, RepeatVector, Dense, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

stocks = 468  # 日資料, 有S&P500中的468支
sp500 = pd.read_pickle('data/SP500.pickle')
# with open('data/SP500.pickle', 'rb') as file:
#     SP500 = pickle.load(file)

all_stocks = np.zeros((len(sp500), sp500[0].shape[0]))
for index, s in enumerate(sp500):
    if s.Close.isna().sum() > 0:
        print(index, s.Close.isna().sum())
        del_index = index
    all_stocks[index] = s.Close.values

all_stocks = np.delete(all_stocks, del_index, axis=0)
scaler = MinMaxScaler()
all_stocks = scaler.fit_transform(all_stocks)

latent_dim = 2
timesteps = sp500[0].shape[0]


def plot_examples(stock_input, stock_decoded):
    n = 10
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, 10))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)


def lstm():
    # AutoEncoder
    inputs = Input(shape=(1, timesteps))
    encoded = CuDNNLSTM(latent_dim)(inputs)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = CuDNNLSTM(latent_dim, return_sequences=True)(decoded)

    sequence_autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    sequence_autoencoder.fit(all_stocks, all_stocks, epochs=20, verbose=2)


# 不合理
def dnn(dim=10):
    # https://morvanzhou.github.io/tutorials/machine-learning/keras/2-6-autoencoder/
    input_img = Input(shape=(timesteps,))
    # encoder layers
    encoded = Dense(64, activation='relu')(input_img)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(dim, activation='relu')(encoded)
    encoder_output = Dense(latent_dim)(encoded)

    # decoder layers
    decoded = Dense(dim, activation='relu')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(timesteps, activation='relu')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)
    # construct the encoder model for plotting
    encoder = Model(input=input_img, output=encoder_output)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(all_stocks, all_stocks, epochs=20, verbose=2)
    return encoder, autoencoder


encoder, autoencoder = dnn(dim=100)

# plotting
stocks_encode = autoencoder.predict(all_stocks)
plot_examples(all_stocks[:, :10], stocks_encode[:, :10])
plt.show()

encoded_imgs = encoder.predict(all_stocks)
# todo: t-SNE
pca = PCA(n_components=2)
encoded_imgs_pca = pca.fit_transform(encoded_imgs)
plt.scatter(encoded_imgs_pca[:, 0], encoded_imgs_pca[:, 1])
plt.xticks(())
plt.yticks(())
plt.show()
# Plot AE embedding with tSNE and plotly
