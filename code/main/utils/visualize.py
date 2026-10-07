import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import torchvision.datasets as datasets

colors = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
    '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BDCF',
    '#FF1493', '#A020F0', '#FF82AB', '#3A5FCD', '#045204',
    '#DDC0A0', '#4EEE94', '#CAFF70', '#FFB90F', '#CD9B9B', ]


def tSNE_visualize(feature, centroids, label, output_name='visual.png'):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    print("processing t-SNE...")
    X = np.vstack((feature, centroids))
    label = np.hstack((label, -1 * np.arange(0, centroids.shape[0], dtype=np.int64) - 1))
    X_embed = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(X)
    print("plotting... ")
    plot_embedding(X_embed, label, "t-SNE", axs)
    plt.savefig(output_name)
    plt.close()
    print('figure saved', output_name)


def tSNE_visualize_both(x, c, rec_x, label, output_name='visual.png'):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    print("processing t-SNE...")

    X = np.vstack((x, c))
    label = np.hstack((label, -1 * np.arange(0, c.shape[0], dtype=np.int64) - 1))
    X_embed = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(X)

    X_r = np.vstack((rec_x, c))
    X_r_embed = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto').fit_transform(X_r)

    print("plotting")
    plot_embedding(X_embed, label, "t-SNE X", axs[0])
    plot_embedding(X_r_embed, label, "t-SNE Denoising X", axs[1])

    plt.savefig(output_name)
    print('figure saved', output_name)
    plt.close()


def tSNE_visualize_steps(x, c, denoise_list, label, output_name='visual.png'):
    fig, axs = plt.subplots(1, len(denoise_list) + 1, figsize=(11 * (len(denoise_list) + 1), 10))
    print("processing t-SNE...")
    X = np.vstack((x, c))
    label = np.hstack((label, -1 * np.arange(0, c.shape[0], dtype=np.int64) - 1))
    scale = X.shape[1] != 2
    if not scale:
        X_embed = X
    else:
        X_embed = TSNE(n_components=2, perplexity=75, init='pca', learning_rate='auto').fit_transform(X)
    plot_embedding(X_embed, label, "t-SNE X", axs[0], scale)

    for i in range(len(denoise_list)):
        X_d = np.vstack((denoise_list[i], c))
        if not scale:
            X_d_embed = X_d
        else:
            X_d_embed = TSNE(n_components=2, perplexity=75, init='pca', learning_rate='auto').fit_transform(X_d)
        plot_embedding(X_d_embed, label, "Denoising step {} X".format(i + 1), axs[i + 1], scale)

    plt.savefig(output_name)
    print('figure saved', output_name)
    plt.close()


def plot_embedding(data, label, title, ax, scale=True):
    if scale:
        X = MinMaxScaler().fit_transform(data)
    else:
        X = data
    for i in range(0, X.shape[0], 1):
        if label[i] < 0:
            true_label = -label[i] - 1
            ax.text(
                X[i, 0],
                X[i, 1],
                str(true_label),
                color='#000000',
                fontdict={'weight': 'bold', 'size': 17},
            )
        else:
            size, color = 12, colors[min(label[i], 19)]
            ax.scatter(X[i, 0], X[i, 1], size, c=color)
        ax.set_title(title)


if __name__ == '__main__':
    datasets.STL10(root='~/dataset/STL10', split='train', transform=None)
    datasets.STL10(root='~/dataset/STL10', split='test', transform=None)
