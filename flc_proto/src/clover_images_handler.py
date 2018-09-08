import numpy as np
from image_utils import normalize


# 教師あり学習用の画像データセットクラス
#   images: 画像データの輝度情報配列
#   labels: 正解ラベル
class ImageDataSet(object):

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    # データセットから”次のバッチサイズ分”のデータを返す
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # 初回のシャッフル
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        # 次のエポックへ
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1

            # このエポックにおいてまだ未選択のデータを取得
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch
                images_new_part = self._images[start:end]
                labels_new_part = self._labels[start:end]
                return np.concatenate((images_rest_part, images_new_part), axis=0), \
                       np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def get_clover_dataset(path, size):
    '''
    クローバー画像を ImageDataSet オブジェクトとして取得する

    :param path: positive/negative フォルダを含む保存先パス
    :param size: モデルに渡す画像ファイルのサイズ（ピクセル）
    :return: ImageDataSet オブジェクト
    '''
    pos_images = np.array([np.asarray(img).ravel() for img in normalize(path + '/positive', size)])
    pos_labels = np.full(shape=pos_images.shape[0], fill_value=1)

    neg_images = np.array([np.asarray(img).ravel() for img in normalize(path + '/negative', size)])
    neg_labels = np.full(shape=neg_images.shape[0], fill_value=0)

    images = np.concatenate((pos_images, neg_images))
    labels = np.concatenate((pos_labels, neg_labels)).reshape(-1, 1)

    return ImageDataSet(images, labels)


def load_clover(path, size=32, training=False):
    '''
    クローバー画像のロード用メソッド

    :param path: positive/negative フォルダを含む保存先パス
    :param size: モデルに渡す画像ファイルのサイズ（ピクセル）
    :param training: True なら訓練用データとしてロードする
    :return: ImageDataSet オブジェクト
    '''
    dataset = get_clover_dataset(path, size)

    if training:
        from sklearn.model_selection import train_test_split
        from collections import namedtuple

        train_images, test_images, train_labels, test_labels = train_test_split(dataset.images, dataset.labels)
        train = ImageDataSet(train_images, train_labels)
        test = ImageDataSet(test_images, test_labels)

        # 訓練用データの場合、学習用とテスト用のデータに分割して扱えるようにする
        Datasets = namedtuple('Datasets', ['train', 'test'])
        return Datasets(train=train, test=test)
    else:
        return dataset
