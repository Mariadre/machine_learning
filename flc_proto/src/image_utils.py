import os
import glob
from PIL import Image


''' ”お行儀よく”使うこと '''


# 変換後の画像ファイルの拡張子
EXTENSION = 'png'


def normalize(path, size, gray_scale=True):
    '''
    モデルに渡す画像ファイルを標準化する
    画像を指定したピクセル数でリサイズし、グレースケールに変換する

    :param path: 変換したい画像ファイルの格納先
    :param size: 変換後のサイズ（size * size に変換）
    :param gray_scale: True ならグレースケールに変換する
    :return: 変換後の画像ファイルリスト
    '''
    files = glob.glob('{}/*.{}'.format(path, EXTENSION))
    mod_files = []
    for file in files:
        img = Image.open(file)
        img_resize = img.resize((size, size))
        if gray_scale:
            img_resize = img_resize.convert('L')
        mod_files.append(img_resize)

    return mod_files


def rename_files_sequentially(path, prefix):
    '''
    画像ファイルを連番でリネームし、PNG形式に変換する
    JPEG 形式に変換させる場合はカラーモードの指定が必要

    :param path: リネームしたい画像ファイルの格納先
    :param prefix: リネーム後のファイル名のプレフィクス
    '''
    extensions = ['jpg', 'jpeg', 'png', 'bmp']
    files = [file for ext in extensions for file in glob.glob(path + '/*.' + ext)]

    for i, f in enumerate(files):
        os.rename(f, '{}/{}_{:03d}.{}'.format(path, prefix, i, EXTENSION))
