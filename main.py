import numpy as np
import tensorflow as tf
import os
import cnn_autoEncoder
from matplotlib import pyplot as plt
import pygame_gen_faces
import pygame_draw_faces


def load_arrays(npy_path: str):
    '''
    Loads in numpy arrays

    Parameters:
    - npy_path (str): Path to the folder where all npy files are stored

    Rasies:
    - ValueError: File path does not exist

    Returns:
    - X_train (4D array): Grayscale (examples, height, width, 1)
    - X_dev (4D array): Grayscale (examples, height, width, 1)

    Note:
    - Output arrays are normalized and of type float16 
        (ie. values range from 0-1)

    '''
    if not os.path.exists(npy_path):
        raise ValueError(f"File path '{npy_path}' does not exist")

    X_train, X_dev = np.load(
        npy_path + "/X_train.npy"), np.load(npy_path + "/X_dev.npy")

    assert X_train.shape == (30_000, 171, 186, 1)
    assert X_dev.shape == (7_921, 171, 186, 1)

    return X_train, X_dev


def learn(X_train, X_dev, auto_path, init=True):
    '''
    Trains autoencoder

    Parameters:
    - X_train (4D array): Data to be trained on (Examples, height, width, channels)
    - auto_path (str): File path to the auto-encoder model 
    - init (bool): initialises new weights and biases if set to True

    Specifications:
    - MSE
    '''
    if init:
        auto_encoder = cnn_autoEncoder.build(X_train.shape[1:])
    else:
        auto_encoder = tf.keras.models.load_model(auto_path, compile=False)

    _, outp_h, outp_w, _ = auto_encoder.layers[-1].output.shape

    start_h = (X_train.shape[1] - outp_h) // 2
    end_h = start_h + outp_h
    start_w = (X_train.shape[2] - outp_w) // 2
    end_w = start_w + outp_w

    auto_encoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                         loss=tf.keras.losses.MeanSquaredError())

    print("Model output shape:", auto_encoder.output.shape)
    print("Training target shape:", X_train[:, start_h:end_h, start_w:end_w, :].shape)

    auto_encoder.fit(
        x=X_train, y=X_train[:, start_h:end_h, start_w:end_w, :],
        batch_size=4,
        epochs=3,
        validation_data=(X_dev, X_dev[:, start_h:end_h, start_w:end_w, :]),
    )

    auto_encoder.save(auto_path, save_format="keras")


def test_auto(auto_path, X):
    '''
    Parameters:
    - auto_path (str): file path to the autoencoder model
    - X (4D array): (Examples, height, width, channels)
    '''
    auto_encoder = tf.keras.models.load_model(auto_path, compile=False)
    reconstruction = auto_encoder.predict(X, verbose=0)
    plot_imgs((X, reconstruction), ("Original", "Reconstruction"))


def plot_imgs(data: tuple, titles: tuple):
    '''
    Produces grid of images from given numpy arrays

    Parameters:
    - data (tuple): List of 4D arrays (batch size, height, width, channel)
        Each array being an image to be displayed
    - titles (tuple): List of titles associated with each batch of arrays

    Raises:
    - ValueError: If number of batches don't match up with number of titles given

    Note:
    - Every batch given opens its own window
    '''
    if len(data) != len(titles):
        raise ValueError

    plt.style.use("dark_background")
    for idx in range(len(data)):
        num_examples = data[idx].shape[0]
        grid_size = int(np.ceil(np.sqrt(num_examples)))
        fig, sub_axes = plt.subplots(grid_size, grid_size,
                                     sharex=True, sharey=True,
                                     figsize=(7, 7))
        fig.suptitle(titles[idx],
                     size="xx-large",
                     weight="bold",
                     y=0.95)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        for ax in sub_axes.ravel():
            ax.set_axis_off()

        example_idx, inner_break = 0, False
        for y in range(grid_size):
            if inner_break:
                break
            for x in range(grid_size):
                try:
                    sub_axes[y, x].imshow(
                        data[idx][example_idx], cmap='gray')
                except IndexError:
                    inner_break = True
                    break
                example_idx += 1
    plt.show()


def extract_decoder(model):
    decoder_inp = model.get_layer("Decoder-Start").input
    decoder_outp = model.get_layer("Decoder-End").output
    return tf.keras.models.Model(decoder_inp, decoder_outp, name="decoder")


def extract_encoder(model):
    ecoder_inp = model.get_layer("Encoder-Start").input
    ecoder_outp = model.get_layer("Encoder-End").output
    return tf.keras.models.Model(ecoder_inp, ecoder_outp, name="encoder")


def latent_space_inference(path, X, num_plots=None):
    '''
    Gains insight into the properties of the latent vectors

    Parameters:
    - path (str): file path to the model
    - X (4D array): (Examples, 28, 28, 1)
    - num_plots (int): Determines the Number of node distributions to be shown,
        Will show all non-zero nodes if not specified

    Raises:
    - ValueError: If file path does not exist

    Note:
    - Not only is a graph that displays the distributions of each node produced
        but also a printed table in terminal output

    Evaluation:
    - Distributions behave somewhat nicely, with a mean of around 2 and deviation of 1.
    - Observing 28 non activated nodes might be concerning but I'm not sure

    Note:
    - These evaluations vary alot if were to train the system again. 
        (As in completely different)
    - For image generation I am going to use total activation of a node across all examples
        as the measure of influence in the output. Will see if this is a good idea or 
        not later I suppose. 
    '''

    if not os.path.exists(path):
        raise ValueError(f"File path '{path}' does not exist")

    auto = tf.keras.models.load_model(path, compile=False)
    encoder = extract_encoder(auto)
    latent = encoder.predict(X, verbose=0)

    total_activation = np.sum(latent, axis=0)
    non_zero_nodes = np.where(total_activation != 0)[0]

    lo, hi, mean, sd = np.min(latent, axis=0), np.max(latent, axis=0), \
        np.mean(latent, axis=0), np.std(latent, axis=0)

    if num_plots is None:
        num_plots = non_zero_nodes.size
    elif num_plots > non_zero_nodes.size:
        raise ValueError(
            f"Number of plots given '{num_plots}' is greater than the number of non-zero output nodes: {non_zero_nodes.size}")

    # method grabbed from internet to get bin size
    bin_num = np.ceil(np.sqrt(latent.shape[0])).astype(np.uint8)
    grid_size = int(np.ceil(np.sqrt(num_plots)))

    plt.style.use("Solarize_Light2")
    fig, sub_axes = plt.subplots(grid_size, grid_size,
                                 sharex=True, sharey=False,
                                 figsize=(7, 7))
    fig.suptitle("Distribution of values for each node",
                 size="xx-large",
                 weight="bold",
                 y=0.95)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85,
                        hspace=0.4)

    lowest_lo, highest_hi = np.min(lo), np.max(hi)
    node_idx, inner_break = 0, False
    for y in range(grid_size):
        if inner_break:  # This method exists both loops
            break
        for x in range(grid_size):
            # if we index something out of range then there is no more nodes to plot
            try:
                # we are using the non-zero-nodes as our index
                sub_axes[y, x].hist(
                    latent[:, non_zero_nodes[node_idx]], bins=bin_num)
                sub_axes[y, x].set(xlim=(lowest_lo, highest_hi))
                sub_axes[y, x].set_title(f"Node {non_zero_nodes[node_idx]+1}",
                                         size="medium")
            except IndexError:
                inner_break = True
                break
            node_idx += 1

    # turn of axis where nothing is being plotted
    for ax in sub_axes.ravel()[node_idx:]:
        ax.set_axis_off()

    print("|{:^8}|{:^8}|{:^8}|{:^8}|{:^12}|".format(
        "Node", "Max", "Mean", "dev", "Total Acts"))
    print("--------------------------------------------------")
    for idx_node in range(latent.shape[1]):
        print("|{:^8}|{:^8}|{:^8}|{:^8}|{:^12}|".format(
            str(idx_node+1),
            str(np.round(hi[idx_node], 2)),
            str(np.round(mean[idx_node], 2)),
            str(np.round(sd[idx_node], 2)),
            str(np.round(total_activation[idx_node], 2))))

    avg_mean = np.mean(mean[non_zero_nodes])
    avg_sd = np.round(np.mean(sd[non_zero_nodes]), decimals=2)

    # Assures me that deviation is not big
    assert np.std(mean[non_zero_nodes]) < 1
    assert np.std(sd[non_zero_nodes]) < 1

    num_zero_nodes = latent.shape[1] - non_zero_nodes.size
    print(f"\nNumber of zero nodes = {num_zero_nodes}")
    print("Mean across all non-zero nodes = {:.2f}".format(avg_mean))
    print("Deviation across all non-zero nodes = {:.2f}".format(avg_sd))
    print("Maximum value across all nodes = {:.2f}".format(highest_hi))
    print("Finally, the deviation of both mean and devation across all non-zero nodes is less than 1\n")

    plt.show()


def generate_faces(path, X):
    '''
    Runs pygame UI
    '''
    auto = tf.keras.models.load_model(path, compile=False)
    decoder = extract_decoder(auto)
    encoder = extract_encoder(auto)
    latent = encoder.predict(X, verbose=0)

    pygame_gen_faces.run(decoder, latent_vectors=latent)


def draw_faces(path):
    auto = tf.keras.models.load_model(path, compile=False)
    pygame_draw_faces.main(auto)
    pass


def print_faces(X, idx_start, idx_end):
    plot_imgs((X[idx_start:idx_end],), ("faces",))

def main():

    # ---------------- Bad Models ---------------- #
    # auto_path = "assets/models/auto-small-data"
    # auto_path = "assets/models/auto-small-data2.1"
    # auto_path = "assets/models/auto-small-selected-data"

    # ---------------- Good Models ---------------- #
    # auto_path = "assets/models/auto-small-data2.2" # For Playing With Sliders
    auto_path = "assets/models/auto2" # For Drawing Faces

    # ---------------- Import Data ---------------- #
    X_train, X_dev = load_arrays("assets/data/numpy")

    # ------- Hand Picked Faces To Train On ---------------- #
    target_indices = [1, 8, 4, 3, 13, 17, 19, 20, 24, 28, 29, 33, 34, 37, 40, 41, 44, 45, 47, 48, 49, 55, 
        58, 59, 61, 64, 67, 70, 72, 75, 77, 79, 83, 84, 85, 86, 91, 96, 97, 100, 101, 102, 103, 
        106, 109, 110, 112, 116, 120, 122, 123, 124, 126, 127, 128, 130, 132, 134, 136, 137, 138, 
        141, 144, 145, 147, 148, 149, 152, 155, 156, 157, 158, 163, 165, 167, 169, 170, 171, 172, 
        174, 177, 178, 181, 182, 184, 185, 188, 189, 190, 191, 192, 193, 194, 195, 198, 200, 201, 
        205, 207, 208, 211, 213, 214, 218, 219, 220, 221, 223, 227, 228, 229, 231, 234, 235, 237, 
        239, 240, 245, 246, 248, 250, 252, 253, 254, 256, 257, 259, 260, 261, 262, 263, 266, 268, 
        269, 270, 271, 273, 278, 279, 281, 282, 283, 285, 286, 290, 291, 292, 293, 294, 295, 297, 
        303, 305, 306, 307, 310, 312, 313, 315, 319, 320, 321, 323, 324, 325, 326, 327, 328, 329, 
        331, 332, 333, 338, 341, 342, 343, 344, 347, 348, 350, 353, 357, 360, 361, 362, 363, 365, 
        367, 368, 370, 375, 377, 379, 381, 382, 383, 384, 385, 386, 387, 390, 391, 392, 393, 394, 
        396, 398, 401, 403, 404, 406, 407, 410, 411, 412, 413, 415, 417, 419, 420, 422, 426, 427, 
        429, 431, 434, 435, 439, 440, 442, 444, 446, 449, 451, 452, 453, 454, 456, 458, 459, 462, 
        463, 467, 468, 471, 472, 473, 474, 479, 481, 482, 490, 494, 495, 496, 499, 502, 504, 505, 
        509, 510, 519, 521, 522, 523, 524, 525, 527, 529, 530, 532, 533, 537, 538, 541, 542, 546, 
        547, 549, 550, 551, 552, 555, 558, 560]
    X_specific = X_train[target_indices]

    # ---------------- Visualise Data ---------------- #
    # print_faces(X_train, idx_start=0, idx_end=16)
    # return


    # ---------------- Training Sequences ---------------- #
    # learn(X_specific, X_dev[3000:3100, :, :, :], auto_path, init=True)
    # learn(X_specific, X_dev[3000:3100, :, :, :], auto_path, init=False)
    # learn(X_train[:500, :, :, :], X_dev[3000:3100, :, :, :], auto_path, init=False)
    # learn(X_train, X_dev[3000:4000, :, :, :], auto_path, init=True)
    # learn(X_train, X_dev[3000:4000, :, :, :], auto_path, init=True)


    # ---------------- Visualise Model Results ---------------- #
    # test_auto(auto_path, X_dev[50:75])

    # ---------------- Latent Variable Inference ---------------- #
    # latent_space_inference(auto_path, X_dev, num_plots=25)

    # ---------------- Generate Faces w/ Model ---------------- #
    # latent_space_inference(auto_path, X_dev, num_plots=25)
    # Generate faces with different input data
    # generate_faces(auto_path, X_dev[:500])
    # generate_faces(auto_path, X_train[:500])
    generate_faces(auto_path, X_specific)

    # ---------------- Generate Faces w/ Drawing ---------------- #
    # draw_faces(auto_path)


if __name__ == "__main__":
    main()
