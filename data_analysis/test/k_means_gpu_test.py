import torch
import numpy as np
import time
# from tqdm import tqdm
import pandas as pd
from kmeans_pytorch import kmeans
# from module_pytorch_kmeans import kmeans
import matplotlib.pyplot as plt


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def sf_kmeans(matrix, device, dims):
    max_range = 40
    wcss = []
    gpu_speeds = []

    #     print(matrix.shape)
    #     print(matrix)
    #     print('\n')

    # data
    #     data_size, dims, num_clusters = 1000, 2, 3
    #     x = np.random.randn(data_size, dims) / 6
    #     x = torch.from_numpy(x)

    for n_clusters in range(2, max_range + 1):
        a = time.time()

        # kmeans
        cluster_ids_x, cluster_centers, iters = kmeans(
            X=matrix, num_clusters=n_clusters, distance='euclidean', tqdm_flag=False, device=torch.device('cuda:0')
        )
        #         iter_limit=500,

        #         print(cluster_ids_x)
        #         print(cluster_centers)
        #         print('\n')
        dists = torch.empty((0, dims)).to(device)
        for i, sample in enumerate(matrix):
            # 0按行追加扩展， 1按列追加扩展
            id = cluster_ids_x[i]
            dist = torch.mul(sample.to(device) - cluster_centers[id].to(device),
                             sample - cluster_centers[id].to(device))
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))

        print(torch.sum(dists))
        #         print('\r{}'.format(torch.sum(dists)), end='')
        wcss.append(torch.sum(dists))

        b = time.time()
        speed = (b - a) / iters
        gpu_speeds.append(speed)
    #         print('\n')

    #     print(wcss)
    #     print(gpu_speeds)

    #     plt.figure()
    plt.grid(linestyle='-.')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.plot(range(max_range - 1), wcss)
    plt.show()

    plt.figure()
    l2, = plt.plot(range(max_range - 1), gpu_speeds, color='g', label="GPU")
    plt.xlabel("num_features")
    plt.ylabel("speed(s/iter)")
    plt.title("Speed with cuda")
    plt.legend(handles=[l2], labels=['GPU'], loc='best')


if __name__ == "__main__":
    # dir_in = r"/home/work/codebook_train_data/"
    data = pd.read_csv('C:\\Users\\19295\PycharmProjects\\Abstract-CTD3-main\\data_analysis\\acc_td3\\td3_risk_acc_logs.csv')
    dims = 8

    df = data.iloc[:, 0:dims]  # get low 16 values
    #     print(df.dtypes)
    #     print(type(data))
    np_data = df.values

    #     data = pd.read_csv(dir_in + 'Mall_Customers.csv')
    #     np_data = data.iloc[1 : 6, [3, 4]].values

    device = choose_device(True)
    matrix = torch.from_numpy(np_data).to(device)
    matrix = matrix.float()

    #     matrix = torch.rand((10000, 10)).to(device)

    sf_kmeans(matrix, device, dims)