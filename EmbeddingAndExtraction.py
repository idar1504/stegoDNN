import hashlib
import cv2
import numpy as np


def sha1_hash_text(input_data):
    """
    Преобразует битовую строку, хешируя ее методом SHA-1
    :param input_data: Битовая последовательность (str)
    :return: Захешированная входная последовательность в 16СС (str)
    """
    sha1 = hashlib.sha1()
    sha1.update(input_data.encode())
    return sha1.hexdigest()


def sha1_hash_array(arr):
    """
    Преобразует массив, хешируя его методом SHA-1
    :param arr: Массив
    :return: Захешированный массив в 16СС (str)
    """
    sha1_hash = hashlib.sha1()
    sha1_hash.update(arr)
    hashed_image = sha1_hash.hexdigest()
    return hashed_image


def fromWatermarkToMessage(path):
    """
    Генерирует из водяного знака сообщение
    :param path: Адрес водяного знака (str)
    :return: Сообщение в виде битовой последовательности (str)
    """
    if isinstance(path, str):
        wm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        wm = path
    wm[wm != 0] = 1
    msg = ''
    for i in wm:
        for j in i:
            msg += str(j)
    return msg


def fromMessageToArray(text, shape):
    """
    Из сообщения формирует массив размером shape
    :param text: Битовая последовательность (str)
    :param shape: Размер выходного массива (int, int)
    :return:
    """
    wm = np.zeros(shape, dtype=np.uint8)
    ind = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if ind < len(text):
                if text[ind] == '1':
                    wm[i, j] = 255
                ind += 1
            else:
                return wm
    return wm


def initial_condition_for_logistic_map(HS):
    """
    Генерирует начальные значения x0 и r для логистической карты
    :param HS: Двоичная хеш последовательность (str)
    :return: Начальные значения логистической карты (int, int)
    """
    while len(HS) < 160:
        HS = '0' + HS
    s1 = sum(int(HS[i]) * (2 ** (40 - i)) for i in range(40))
    s2 = sum(int(HS[i]) * (2 ** (80 - i)) for i in range(40, 80))
    t1 = sum(int(HS[i]) * (2 ** (120 - i)) for i in range(80, 120))
    t2 = sum(int(HS[i]) * (2 ** (160 - i)) for i in range(120, 160))

    x0 = s1 / (2 ** 40)
    r0 = s2 / (2 ** 40)

    x1 = (x0 * s1 * t1 / (2 ** 40) + x0) % 1
    r1 = (r0 * s1 * t1 / (2 ** 40) + r0) % 4
    x2 = (x1 * s2 * t2 / (2 ** 40) + x1) % 1
    r2 = (r1 * s2 * t2 / (2 ** 40) + r1) % 4

    return x2, 4 - r2


def generate_random_sequence(key, message):
    """
    Из ключа и сообщения генерируется двоичную хэш последовательность
    :param key: Ключ (str)
    :param message: Сообщение (str или array)
    :return: Хэш последовательность (str)
    """
    if isinstance(message, str):
        hash_sequence_message = sha1_hash_text(message)
    else:
        hash_sequence_message = sha1_hash_array(message)
    hash_sequence_user = sha1_hash_text(key)
    random_sequence = int(hash_sequence_user, 16) ^ int(hash_sequence_message, 16)
    return random_sequence


def divide_into_blocks(array, size):
    """
    Входной массив разделяется на список массивов размером size*size
    :param array: Массив
    :param size: Размер блоков (int)
    :return: Список блоков размером size*size (list)
    """
    height, width = array.shape
    blocks = []
    for y in range(0, height, size):
        for x in range(0, width, size):
            block = array[y:y + size, x:x + size]
            b = []
            for i in block:
                row = []
                for j in i:
                    row.append(int(j))
                b.append(row)
            blocks.append(b)
    return blocks


def place_blocks(blocks, mn, size):
    """
    Размещает блоки в единый массив
    :param blocks: Список блоков (list)
    :param mn: Размер итогового массива (int, int)
    :param size: Размер блоков (int)
    :return: Массив размеров mn
    """
    m, n = mn
    if m % size != 0 or n % size != 0:
        print("Ошибка: Размеры m и n должны быть кратны ", str(size))
        return None
    new_array = np.zeros((m, n), dtype=int)
    row, col = 0, 0
    for block in blocks:
        new_array[row:row + size, col:col + size] = block
        col += size
        if col >= n:
            col = 0
            row += size
    return new_array


def error_computing_first_phase(block):
    """
    Расчет ошибок первой фазы (второй для извлечения)
    :param block: Блок (list)
    :return: Список ошибок для соответствующих элементов блока (list)
    """
    central_pixel = block[1][1]
    errors = [block[0][1] - central_pixel,
              block[1][0] - central_pixel,
              block[1][2] - central_pixel,
              block[2][1] - central_pixel]
    return errors


def error_computing_second_phase(block):
    """
    Расчет ошибок второй фазы (первой для извлечения)
    :param block: Блок (list)
    :return: Список ошибок для соответствующих элементов блока (list)
    """
    central_pixel = block[1][1]
    errors = [int(block[0][0] - int(np.mean([block[0][1], block[1][0], central_pixel]))),
              int(block[0][2] - int(np.mean([block[0][1], block[1][2], central_pixel]))),
              int(block[2][0] - int(np.mean([block[1][0], block[2][1], central_pixel]))),
              int(block[2][2] - int(np.mean([block[1][2], block[2][1], central_pixel])))]
    return errors


def count_values(array, value):
    """
    Считает количество значения в двумерном массиве
    :param array: Входной массив
    :param value: Значение
    :return: Количество значения в массиве (int)
    """
    count = 0
    for row in array:
        for element in row:
            count += 1 if element == value else 0
    return count


def array_to_string(array):
    """
    Генерирует строку из двумерного массива
    :param array: Двумерный массив
    :return: Строка элементов массива (str)
    """
    result = ""
    for row in array:
        for element in row:
            result += str(element)
    return result


def good(block, action):
    """
    Сжимает блок Location map в строку
    :param block: Блок LM
    :param action: Действие, определяющее, относительно какого значения происходит сжатие (int)
    :return: Сжатый блок LM (str)
    """
    block_array = np.array(block).flatten()
    compressed_info = []
    prev_index = -1
    count = 0
    for i, value in enumerate(block_array):
        if value == action:
            if prev_index == -1:
                binary_index = format(i + 1, 'b').zfill(int(np.ceil(np.log2(len(block_array)))))
            else:
                binary_index = format(i - prev_index, 'b').zfill(
                    max(int(np.ceil(np.log2(len(block_array) - prev_index))), 1))
            compressed_info.append(binary_index)
            prev_index = i
            count += 1

    compressed_string = format(count, 'b') + ' ' + ' '.join(compressed_info)
    return compressed_string


def compress_location_map(LM):
    """
    Сжимает Location map
    :param LM: Location map
    :return: Сжатая LM (list)
    """
    locationMap = divide_into_blocks(LM, 5)
    compressedLM = []
    T = 4
    for i in locationMap:
        b0 = count_values(i, 0)
        b1 = count_values(i, 1)
        if b0 == 0:
            compressedLM.append('11')
            continue
        elif b1 == 0:
            compressedLM.append('10')
            continue
        M = min(b0, b1)
        if 1 <= M <= T and b0 < b1:
            toAdd = good(i, 0)
            compressedLM.append('011 ' + toAdd)
        elif 1 <= M <= T and b1 < b0:
            toAdd = good(i, 1)
            compressedLM.append('010 ' + toAdd)
        else:
            compressedLM.append('00 ' + array_to_string(i))
    return compressedLM


def fromGood(compressedBlock, value):
    """
    Восстанавливает блок Location map из сжатого варианта
    :param compressedBlock: Сжатый блок LM
    :param value: Действие, определяющее, относительно какого значения происходит сжатие (int)
    :return: Блок LM
    """
    compressedBlock = compressedBlock.split()[1:]
    compressedBlock = [int(compressedBlock, 2) for compressedBlock in compressedBlock]
    n = compressedBlock[0]
    z = compressedBlock[1:]
    arr = np.ones(25, dtype=int)
    idx = -1
    for i in range(n):
        idx += z[i]
        arr[idx] = value
    block = np.reshape(arr, (5, 5))
    return block


def fromBad(compressedBlock):
    """
    Восстанавливает блок Location map со смешанными 0 и 1 из сжатого варианта
    :param compressedBlock: Сжатый блок LM
    :return: Блок LM
    """
    compressedBlock = compressedBlock.split()[1]
    size = len(compressedBlock) // 5
    block = np.zeros((size, 5), dtype=int)
    for i in range(size):
        for j in range(5):
            block[i, j] = int(compressedBlock[i * 5 + j])
    return block


def decompress_location_map(compressedLM, size):
    """
    Восстанавливает Location map из сжатого варианта
    :param compressedLM: Сжатая LM (list)
    :param size: Размер LM
    :return: Восстановленная LM
    """
    m, n = size
    LM = []
    for i in compressedLM:
        if i == '11':
            toAdd = np.ones((5, 5), dtype=int)
            LM.append(toAdd)
        elif i == '10':
            toAdd = np.zeros((5, 5), dtype=int)
            LM.append(toAdd)
        elif i.split()[0] == '011':
            toAdd = fromGood(i, 0)
            LM.append(toAdd)
        elif i.split()[0] == '010':
            toAdd = fromGood(i, 1)
            LM.append(toAdd)
        elif i.split()[0] == '00':
            toAdd = fromBad(i)
            LM.append(toAdd)
        else:
            return None
    LM = np.array(LM)
    LM = place_blocks(LM, (m, n), 5)
    return LM


def generateZ(x0, r, size):
    """
    Генерирует двумерный массив Z из начальных x0 и r
    :param x0: Начальное значение x
    :param r: Параметр r
    :param size: Размерность массива
    :return: Массив Z
    """
    m, n = size
    x = x0
    Z = np.zeros((m, n), dtype=int)
    for i in range(m):
        for j in range(n):
            zi = int(x * (2 ** 50)) % 256
            Z[i][j] = zi
            x = r * x * (1 - x)
    return Z


def data_embedding(CI, message, base, data_key, image_key):
    """
    Встраивает сообщение (водяной знак) в cover image и генерирует Location map
    :param CI: Cover image (массив)
    :param message: Встраиваемое сообщение (водяной знак) (str)
    :param base: Основание встраиваемого сообщения (int)
    :param data_key: Ключ для шифрования сообщения (str)
    :param image_key: Ключ для шифрования изображения (str)
    :return: Стего-изображение, сжатая LM, кортеж с ключами для извлечения (tuple)
    """
    blocks = divide_into_blocks(CI, 3)
    P = np.copy(blocks)
    m, n = CI.shape[:2]
    LM = divide_into_blocks(np.zeros((m, n), dtype=int), 3)

    data_key_ = generate_random_sequence(data_key, message)
    x0, r = initial_condition_for_logistic_map(bin(data_key_)[2:])
    w = ''
    x = x0
    for i in message:
        wi = round(x) ^ int(i, base)
        w += str(wi)
        x = r * x * (1 - x)
    w_cnt = 0

    for i in range(len(blocks)):
        if w_cnt >= len(w):
            break

        errors1 = error_computing_first_phase(P[i])

        if w_cnt < len(w):
            new_p = blocks[i][0][1] + (2 ** (base - 1) - 1) * errors1[0] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][0][1] = new_p
                LM[i][0][1] = 1
                w_cnt += 1

        if w_cnt < len(w):
            new_p = blocks[i][1][0] + (2 ** (base - 1) - 1) * errors1[1] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][1][0] = new_p
                LM[i][1][0] = 1
                w_cnt += 1

        if w_cnt < len(w):
            new_p = blocks[i][1][2] + (2 ** (base - 1) - 1) * errors1[2] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][1][2] = new_p
                LM[i][1][2] = 1
                w_cnt += 1

        if w_cnt < len(w):
            new_p = blocks[i][2][1] + (2 ** (base - 1) - 1) * errors1[3] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][2][1] = new_p
                LM[i][2][1] = 1
                w_cnt += 1

        errors2 = error_computing_second_phase(P[i])

        if w_cnt < len(w):
            new_p = blocks[i][0][0] + (2 ** (base - 1) - 1) * errors2[0] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][0][0] = new_p
                LM[i][0][0] = 1
                w_cnt += 1

        if w_cnt < len(w):
            new_p = blocks[i][0][2] + (2 ** (base - 1) - 1) * errors2[1] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][0][2] = new_p
                LM[i][0][2] = 1
                w_cnt += 1

        if w_cnt < len(w):
            new_p = blocks[i][2][0] + (2 ** (base - 1) - 1) * errors2[2] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][2][0] = new_p
                LM[i][2][0] = 1
                w_cnt += 1

        if w_cnt < len(w):
            new_p = blocks[i][2][2] + (2 ** (base - 1) - 1) * errors2[3] + int(w[w_cnt])
            if 0 <= new_p <= 255:
                P[i][2][2] = new_p
                LM[i][2][2] = 1
                w_cnt += 1

    SI = place_blocks(np.array(P), (m, n), 3)
    image_key_ = generate_random_sequence(image_key, SI)
    x0, r = initial_condition_for_logistic_map(bin(image_key_)[2:])
    # Z = generateZ(x0, r, (m, n))
    # for i in range(m):
    #     for j in range(n):
    #         SI[i][j] = SI[i][j] ^ int(Z[i][j])
    LM = place_blocks(LM, (m, n), 3)
    compressedLocationMap = compress_location_map(LM)
    keys = [data_key_, image_key_]
    return SI, compressedLocationMap, keys


def data_extraction(SI, CLM, data_key, image_key, base):
    """
    Извлекает из стего-изображения сообщение (водяной знак) и восстанавливает cover image
    :param SI: Стего-изображение (массив)
    :param CLM: Сжатая Location map
    :param data_key: Ключ дешифрования встроенного сообщения
    :param image_key: Ключ дешифрования изображения
    :param base: Основание встроенного сообщение
    :return: CI, сообщение
    """
    m, n = SI.shape[:2]
    LM = divide_into_blocks(decompress_location_map(CLM, (m, m)), 3)
    x0, r = initial_condition_for_logistic_map(bin(image_key)[2:])
    x = x0
    # Z = generateZ(x, r, (m, n))
    # for i in range(m):
    #     for j in range(n):
    #         SI[i][j] = SI[i][j] ^ int(Z[i][j])

    blocks = divide_into_blocks(SI, 3)
    P = np.copy(blocks)
    message = ''
    x0, r = initial_condition_for_logistic_map(bin(data_key)[2:])
    x = x0

    for i in range(len(blocks)):
        w1 = ''
        w2 = ''
        errors1 = error_computing_second_phase(P[i])
        errors2 = error_computing_first_phase(P[i])

        if LM[i][0][1] == 1:
            wi_ = abs(errors2[0] - (2 ** (base - 1)) * int(errors2[0] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w2 += str(wi)
            x = r * x * (1 - x)
            P[i][0][1] = blocks[i][0][1] - (2 ** (base - 1) - 1) * int(errors2[0] / (2 ** (base - 1))) - wi
        if LM[i][1][0] == 1:
            wi_ = abs(errors2[1] - (2 ** (base - 1)) * int(errors2[1] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w2 += str(wi)
            x = r * x * (1 - x)
            P[i][1][0] = blocks[i][1][0] - (2 ** (base - 1) - 1) * int(errors2[1] / (2 ** (base - 1))) - wi
        if LM[i][1][2] == 1:
            wi_ = abs(errors2[2] - (2 ** (base - 1)) * int(errors2[2] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w2 += str(wi)
            x = r * x * (1 - x)
            P[i][1][2] = blocks[i][1][2] - (2 ** (base - 1) - 1) * int(errors2[2] / (2 ** (base - 1))) - wi
        if LM[i][2][1] == 1:
            wi_ = abs(errors2[3] - (2 ** (base - 1)) * int(errors2[3] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w2 += str(wi)
            x = r * x * (1 - x)
            P[i][2][1] = blocks[i][2][1] - (2 ** (base - 1) - 1) * int(errors2[3] / (2 ** (base - 1))) - wi

        if LM[i][0][0] == 1:
            wi_ = abs(errors1[0] - (2 ** (base - 1)) * int(errors1[0] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w1 += str(wi)
            x = r * x * (1 - x)
            P[i][0][0] = blocks[i][0][0] - (2 ** (base - 1) - 1) * int(errors1[0] / (2 ** (base - 1))) - wi
        if LM[i][0][2] == 1:
            wi_ = abs(errors1[1] - (2 ** (base - 1)) * int(errors1[1] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w1 += str(wi)
            x = r * x * (1 - x)
            P[i][0][2] = blocks[i][0][2] - (2 ** (base - 1) - 1) * int(errors1[1] / (2 ** (base - 1))) - wi
        if LM[i][2][0] == 1:
            wi_ = abs(errors1[2] - (2 ** (base - 1)) * int(errors1[2] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w1 += str(wi)
            x = r * x * (1 - x)
            P[i][2][0] = blocks[i][2][0] - (2 ** (base - 1) - 1) * int(errors1[2] / (2 ** (base - 1))) - wi
        if LM[i][2][2] == 1:
            wi_ = abs(errors1[3] - (2 ** (base - 1)) * int(errors1[3] / (2 ** (base - 1))))
            wi = round(x) ^ wi_
            w1 += str(wi)
            x = r * x * (1 - x)
            P[i][2][2] = blocks[i][2][2] - (2 ** (base - 1) - 1) * int(errors1[3] / (2 ** (base - 1))) - wi

        message += (w2 + w1)

    CI = place_blocks(P, (m, n), 3)
    return CI, message
