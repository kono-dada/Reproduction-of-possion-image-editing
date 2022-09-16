from cv2 import cv2
import numpy as np


def delta_with_mixed_gradient(h: int, w: int):
    global g, back
    return sum([(g[_] - g[h, w] if ((g[_] - g[h, w]) ** 2).sum() > ((back[_] - back[h, w]) ** 2).sum()
                 else back[_] - back[h, w]) for _ in neighbors(h, w)])


def delta_test_2(h: int, w: int):
    global g, back
    alpha = ((g[h, w] - back[h, w]) ** 2).sum() ** (1 / 2)
    _max = 3 * 256
    return sum([(g[_] - g[h, w]) * alpha / _max + (back[_] - back[h, w]) * (1 - alpha / _max) for _ in neighbors(h, w)])


def delta_test(h: int, w: int):
    global g, back
    return sum([(g[_] - g[h, w]) for _ in neighbors(h, w)]) \
        if ((g[h - 1, w] - g[h + 1, w]) ** 2).sum() + ((g[h, w - 1] - g[h, w + 1]) ** 2).sum() > \
           ((back[h - 1, w] - back[h + 1, w]) ** 2).sum() + ((back[h, w - 1] - back[h, w + 1]) ** 2).sum() \
        else sum([(back[_] - back[h, w]) for _ in neighbors(h, w)])


def extract_back_image():
    _back = np.zeros([g_height, g_width, 3])
    for __h in range(g_height):
        for __w in range(g_width):
            _back[__h, __w] = s[__h + basis[0], __w + basis[1]]
    return _back


def delta(h, w):
    global g
    return sum([g[_] - g[h, w] for _ in neighbors(h, w)])


def neighbors(h, w):
    return (h + 1, w), (h - 1, w), (h, w + 1), (h, w - 1)


if __name__ == '__main__':
    basis = (100, 100)  # the position of the front image on the back image
    undesired = np.array([0, 0, 0]).all()

    g: np.ndarray = cv2.imread('dingzhen.jpeg').astype(np.float64)  # represent g in the original paper
    g_height, g_width, _ = g.shape
    # g_mask: np.ndarray = cv2.imread('trump_mask.jpg')
    g_mask = np.zeros([g_height, g_width, 3])
    g_mask[:, :] = [255, 255, 255]
    s: np.ndarray = cv2.imread('apple.jpg').astype(np.float64)  # represent s in the original paper
    s_height, s_width, _ = s.shape
    back = extract_back_image()
    # mark the main part with numbers by using masks.
    number_of_digits = 0
    series_of_digits = np.zeros([g_height, g_width], np.uint16)
    for _h in range(0, g_height):
        for _w in range(0, g_width):
            if (_h == 0 or _h == g_height - 1) or (_w == 0 or _w == g_width - 1):
                g_mask[_h, _w] = np.array([0, 0, 0])
            if not g_mask[_h, _w].all() == undesired:  # the undesired part is pure black on the mask
                series_of_digits[_h, _w] = number_of_digits
                number_of_digits += 1

    matrix_A = np.zeros([number_of_digits, number_of_digits, 3], np.int8)
    vector_b = np.zeros([number_of_digits, 3])
    for _h in range(g_height):
        for _w in range(g_width):
            # work out b
            if not g_mask[_h, _w].all() == undesired:
                # temp_b = delta_with_mixed_gradient(_h, _w)
                temp_b = delta(_h, _w)
                # temp_b = delta_test(_h, _w)
                # temp_b = delta_test_2(_h, _w)
                for neighbor in neighbors(_h, _w):
                    # if the neighbour digit is on the boundary
                    if g_mask[neighbor].all() == undesired:
                        temp_b -= back[neighbor]
                        # temp_b -= g[digit[0], digit[1]]
                    else:  # if not
                        matrix_A[series_of_digits[_h, _w], series_of_digits[neighbor]] = np.array([1, 1, 1])
                matrix_A[series_of_digits[_h, _w], series_of_digits[_h, _w]] = np.array([-4, -4, -4])
                vector_b[series_of_digits[_h, _w]] = temp_b

    channels = []
    for _ in range(3):
        a_solution = np.linalg.solve(matrix_A[:, :, _], vector_b[:, _])
        a_solution[a_solution < 0] = 0
        a_solution[a_solution > 255] = 255
        i = 0
        one_channel = np.zeros([g_height, g_width])
        for _h in range(g_height):
            for _w in range(g_width):
                if not g_mask[_h, _w].all() == undesired:
                    one_channel[_h, _w] = a_solution[i]
                    i += 1
        channels.append(one_channel)

    final_solution = np.dstack(channels).astype(np.uint8)

    # copy the result onto the back image
    for _h in range(g_height):
        for _w in range(g_width):
            if not g_mask[_h, _w].all() == undesired:
                s[_h + basis[0], _w + basis[1]] = final_solution[_h, _w]
    cv2.imwrite('test2.jpg', s.astype(np.uint8))
    cv2.imshow('1', s.astype(np.uint8))
    cv2.waitKey(0)
