import math
import numpy as np
from scipy.ndimage import affine_transform
from skimage.metrics import structural_similarity as ssim
import EmbeddingAndExtraction as EaE


def bpp(payload, size):
    """
    Оценивает вместимость
    :param payload: Длина встроенного сообщения (int)
    :param size: Размер стего-изображения (int, int)
    :return: Количество встроенных битов на пиксель ()
    """
    M, N = size
    EmbeddingRate = payload / (M * N)
    return EmbeddingRate


def mse(CI, SI, size):
    """
    Оценивает среднеквадратичную ошибку
    :param CI: Cover image
    :param SI: Стего-изображение
    :param size: Размер изображения
    :return: Среднеквадратичная ошибка
    """
    M, N = size
    MSE = 0
    for i in range(M):
        for j in range(N):
            MSE += (CI[i][j] - SI[i][j]) ** 2
    MSE /= M * N
    return MSE


def psnr(CI, SI, size):
    """
    Оценивает незаметность
    :param CI: Cover image
    :param SI: Стего-изображение
    :param size: Размер изображения
    :return: Метрика PSNR
    """
    MSE = mse(CI, SI, size)
    PSNR = 10 * math.log10(255 * 255 / MSE)
    return PSNR


def ber(message, extracted, base):
    """
    Оценивает точность извлечения
    :param message:
    :param extracted:
    :param base:
    :return: Метрика BER, показывающая количество неверно извлеченных битов
    """
    BER = 0
    for i in range(len(message)):
        BER += (int(message[i], base) ^ int(extracted[i], base)) * 100
    BER /= len(message)
    return BER


def SSIM(CI, SI):
    """
    Измеряет SSIM
    :param CI: Cover image
    :param SI: Стего изображение
    :return: Метрика SSIM
    """
    return ssim(CI, SI, data_range=255)


def translation(image, dx, dy):
    """
    Производит атаку translation
    :param image: Изображение
    :param dx: Параметр атаки
    :param dy: Параметр атаки
    :return: Измененное изображение
    """
    translated_image = np.roll(image, (dy, dx), axis=(0, 1))
    return translated_image


def rotation(image, angle):
    """
    Производит атаку rotation
    :param image: Изображение
    :param angle: Угол
    :return: Измененное изображение
    """
    rotated_image = np.rot90(image, k=angle // 90)
    return rotated_image


def reflection(image, axis):
    """
    Производит атаку reflection
    :param image: Изображение
    :param axis: Ось
    :return: Измененное изображение
    """
    reflected_image = np.flip(image, axis=axis)
    return reflected_image


def scaling(image, scale):
    """
    Производит атаку scaling
    :param image: Изображение
    :param scale: Масштаб
    :return: Измененное изображение
    """
    scaled_image = affine_transform(image, matrix=np.diag([scale, scale]), output_shape=image.shape)
    return scaled_image


def authenticationAnalysis(watermark_extracted, watermark_original, base):
    """
    Производит анализ на аутентификацию, выводит ее подробные результаты после каждой атаки
    :param watermark_extracted: Извлеченный водяной знак
    :param watermark_original: Изначальный водяной знак
    :param base: СС
    :return: None
    """
    attacks = ["translation", "rotation", "reflection", "scaling"]
    for attack in attacks:
        if attack == "translation":
            watermark_received_attacked = translation(watermark_extracted, dx=40, dy=40)
        elif attack == "rotation":
            watermark_received_attacked = rotation(watermark_extracted, angle=90)
        elif attack == "reflection":
            watermark_received_attacked = reflection(watermark_extracted, axis=0)
        else:
            watermark_received_attacked = scaling(watermark_extracted, scale=2)
        BER = ber(EaE.fromWatermarkToMessage(watermark_received_attacked),
                  EaE.fromWatermarkToMessage(watermark_original), base)
        print("BER after", attack, "attack:", BER)


def reversibilityAnalysis(image1, image2):
    """
    Строит массив различий между двумя изображениями
    :param image1:
    :param image2:
    :return:
    """
    if len(image1) != len(image2) or len(image1[0]) != len(image2[0]):
        raise ValueError("Размеры изображений не совпадают")

    difference_image = []

    for i in range(len(image1)):
        row = []
        for j in range(len(image1[0])):
            pixel_difference = abs(image1[i][j] - image2[i][j])
            row.append(pixel_difference)
        difference_image.append(row)

    return np.array(difference_image)
