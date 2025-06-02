import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


def load_data(filepath, idx):
    data = []
    valid_idx = idx
    for i in tqdm(range(idx)):
        advector = np.load(filepath + "advector_" + str(i) + ".npy")
        advectee = np.load(filepath + "advectee_" + str(i) + ".npy")
        flux = np.load(filepath + "corrective_flux_" + str(i) + ".npy")
        # if any of the values are nan, skip this stencil
        if np.isnan(advector).any() or np.isnan(advectee).any() or np.isnan(flux).any():
            valid_idx -= 1
            continue
        data.append([advector, advectee, -flux])

    for i in tqdm(range(valid_idx)):
        data[i][1] = pd.DataFrame(data[i][1]).fillna(method="ffill").values
        data[i][0] = pd.DataFrame(data[i][0]).fillna(method="ffill").values
        data[i][2] = pd.DataFrame(data[i][2]).fillna(method="ffill").values
        data[i][1] = pd.DataFrame(data[i][1]).fillna(method="bfill").values
        data[i][0] = pd.DataFrame(data[i][0]).fillna(method="bfill").values
        data[i][2] = pd.DataFrame(data[i][2]).fillna(method="bfill").values
    return data


def split_into_stencils(data, stencil_length, max_advector, max_advectee, max_flux):
    stencils = []
    for i in range(len(data)):
        advector, advectee, flux = data[i]
        for j in range(0, len(data[i][1]) - 1):
            advectee_stencil = advectee[j - 1 : j + 1]
            advector_stencil = advector[j - 1 : j + 2]
            flux_stencil = flux[j]
            # print(f"{advectee_stencil.shape=}, {advector_stencil.shape=}, {flux_stencil.shape=}")
            if (
                np.isnan(advectee_stencil).any()
                or np.isnan(advector_stencil).any()
                or np.isnan(flux_stencil).any()
            ):
                continue
            try:
                assert advectee_stencil.shape[0] == stencil_length - 1
                assert advector_stencil.shape[0] == stencil_length
            except AssertionError:
                # print(f"ERROR: advectee_stencil.shape[0]: {advectee_stencil.shape[0]}, advector_stencil.shape[0]: {advector_stencil.shape[0]}")
                continue
            advector_stencil = advector_stencil / max_advector
            advector_stencil = advector_stencil.flatten()
            advectee_stencil = advectee_stencil / max_advectee
            advectee_stencil = advectee_stencil.flatten()
            flux_stencil = flux_stencil / max_flux
            flux_stencil = flux_stencil.flatten()

            input_stencil = np.concatenate((advector_stencil, advectee_stencil), axis=0)
            stencils.append((input_stencil, flux_stencil[0]))
    return stencils


def get_stencils_2(data, stencil_length, train_intervals, val_intervals):
    # split data length into 20
    basic_interval = len(data) // 10
    train_start_1 = basic_interval * train_intervals[0][0]
    train_end_1 = basic_interval * train_intervals[0][1]
    train_start_2 = basic_interval * train_intervals[1][0]
    train_end_2 = basic_interval * train_intervals[1][1]
    val_start_1 = basic_interval * val_intervals[0][0]
    val_end_1 = basic_interval * val_intervals[0][1]

    data_train_1 = data[train_start_1:train_end_1]
    data_train_2 = data[train_start_2:train_end_2]
    data_val_1 = data[val_start_1:val_end_1]
    print(f"{train_start_1=},{train_end_1=}, {val_start_1=}, {val_end_1=}")
    data_train = data_train_1 + data_train_2
    data_val = data_val_1
    print(f"{len(data_train)=}, {len(data_val)=}")

    max_advector = max(data[:][0].max() for data in data_train)
    max_advectee = max(data[:][1].max() for data in data_train)
    max_flux = max(data[:][2].max() for data in data_train)

    train_stencils = split_into_stencils(
        data_train, stencil_length, max_advector, max_advectee, max_flux
    )
    val_stencils = split_into_stencils(
        data_val, stencil_length, max_advector, max_advectee, max_flux
    )

    return train_stencils, val_stencils


def sanity_check(model, val_data, scaling_factor=1):
    X_test = val_data["input"]
    y_test = val_data["flux"]
    X_test, y_test = np.stack(X_test), np.stack(y_test)
    # calculate the sum of y_test and the prediction
    y_test_sum = np.sum(y_test)
    outputs = model.predict(X_test)
    output = np.concatenate(outputs) * scaling_factor
    # output_sum = np.sum(output)
    # print(f"{y_test_sum=}, {output_sum=}")
    plt.plot(output)
    plt.plot(y_test)
    plt.legend(["output", "y_test"])
    plt.show()


def get_model(n_hidden_layers, n_neurons, n_inputs=5):
    model = keras.Sequential()
    # add the input layer
    model.add(layers.Dense(n_neurons, activation="relu", input_shape=(n_inputs,)))
    for i in range(n_hidden_layers):
        model.add(layers.Dense(n_neurons, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model, df_train, df_test, epochs=100):
    X_train = df_train["input"].values
    y_train = df_train["flux"].values
    X_test = df_test["input"].values
    y_test = df_test["flux"].values
    X_train, y_train = np.stack(X_train), np.stack(y_train)
    X_test, y_test = np.stack(X_test), np.stack(y_test)
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32
    )
    return model, history


def visualize_training(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.yscale("log")
    plt.show()


def visualize_output(model, val_data, scaling_factor=1, label="corrective flux"):
    X_test = val_data["input"]
    y_test = val_data["flux"]
    X_test, y_test = np.stack(X_test), np.stack(y_test)
    outputs = model.predict(X_test)
    output = np.concatenate(outputs) * scaling_factor
    plt.figure(figsize=(6, 6))
    plt.plot(output)
    plt.plot(y_test)
    plt.legend(["output", "y_test"])
    plt.xlabel("x/delta x")
    plt.ylabel("corrective flux")
    plt.grid()
    plt.savefig(f"output_{label}.png")
    plt.show()
