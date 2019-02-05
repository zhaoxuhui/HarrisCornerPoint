# coding=utf-8
import numpy as np
import cv2
import math


def getXWindow(img, i, j, win_size):
    if win_size % 2 == 0:
        win = None
        return win
    half_size = win_size / 2
    start_x = i - half_size - 1
    start_y = j - half_size
    end_x = i + half_size + 1 + 1
    end_y = j + half_size + 1
    win = img[start_y:end_y, start_x:end_x]
    return win


def getYWindow(img, i, j, win_size):
    if win_size % 2 == 0:
        win = None
        return win
    half_size = win_size / 2
    start_x = i - half_size
    start_y = j - half_size - 1
    end_x = i + half_size + 1
    end_y = j + half_size + 1 + 1
    win = img[start_y:end_y, start_x:end_x]
    return win


def getWindowWithRange(img, i, j, win_size):
    if win_size % 2 == 0:
        win = None
        return win
    half_size = win_size / 2
    start_x = i - half_size
    start_y = j - half_size
    end_x = i + half_size + 1
    end_y = j + half_size + 1
    win = img[start_x:end_x, start_y:end_y]
    return win, start_x, end_x, start_y, end_y


def getGaussianFilter(win_size=3, sigma=0.8):
    xs = []
    ys = []
    g_xys = []
    g_xys_normal = []
    g_xys_int = []
    cen_x = win_size / 2
    cen_y = win_size / 2
    for i in range(win_size):
        for j in range(win_size):
            cen_i = i - cen_x
            cen_j = j - cen_y
            g_xy = (1 / (2.0 * math.pi * sigma * sigma)) * \
                   math.pow(math.e, -(cen_i * cen_i + cen_j * cen_j) / (2.0 * sigma * sigma))
            xs.append(i)
            ys.append(j)
            g_xys.append(g_xy)
    scale_num = 1.0 / sum(g_xys)

    for i in range(g_xys.__len__()):
        g_xys_normal.append(g_xys[i] * scale_num)

    filter = np.zeros([win_size, win_size])
    for i in range(win_size):
        for j in range(win_size):
            filter[i, j] = g_xys_normal[i * win_size + j]

    min_num = min(g_xys_normal)
    for i in range(g_xys_normal.__len__()):
        g_xys_int.append(int(g_xys_normal[i] / min_num))

    filter_int = np.zeros([win_size, win_size])
    for i in range(win_size):
        for j in range(win_size):
            filter_int[i, j] = g_xys_int[i * win_size + j]

    return filter, filter_int


def calcIx(win):
    win = np.int32(win)
    size = win.shape[1] - 2
    Ix = np.zeros([size, size], np.int32)
    for j in range(win.shape[0]):
        for i in range(1, win.shape[1] - 1):
            Ix[j, i - 1] = win[j, i + 1] - win[j, i - 1]
    return Ix


def calcIy(win):
    win = np.int32(win)
    size = win.shape[0] - 2
    Iy = np.zeros([size, size], np.int32)
    for j in range(1, win.shape[0] - 1):
        for i in range(win.shape[1]):
            Iy[j - 1, i] = win[j + 1, i] - win[j - 1, i]
    return Iy


def calcCornerResponse(img, i, j, gauss_kernel, win_size=3, k=0.04):
    winX = getXWindow(img, i, j, win_size)
    Ix = calcIx(winX)
    winY = getYWindow(img, i, j, win_size)
    Iy = calcIy(winY)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    A = np.sum((gauss_kernel * Ix2) / np.sum(gauss_kernel))
    B = np.sum((gauss_kernel * Iy2) / np.sum(gauss_kernel))
    C = np.sum((gauss_kernel * IxIy) / np.sum(gauss_kernel))
    det_M = A * B - C * C
    trace_M = A + B
    R = det_M - k * trace_M * trace_M
    return R


def thresholdCornerMap(corner_map, th=100):
    if th == -1:
        th = np.mean(corner_map)
        print 'auto th:', th
    else:
        pass
    corner_map = np.where(corner_map < th, 0, corner_map)
    return corner_map


def nonMaximumSupression(mat, nonMaxValue=0):
    mask = np.zeros(mat.shape, mat.dtype) + nonMaxValue
    max_value = np.max(mat)
    loc = np.where(mat == max_value)
    row = loc[0]
    col = loc[1]
    mask[row, col] = max_value
    return mask, row, col


def getScore(item):
    return item[2]


def getKeypoints(keymap, nonMaxValue, nFeature=-1):
    # 用于获取角点的坐标以及对角点进行排序筛选
    loc = np.where(keymap != nonMaxValue)
    xs = loc[1]
    ys = loc[0]
    print xs.__len__(), 'keypoints were found.'
    kps = []
    for x, y in zip(xs, ys):
        kps.append([x, y, keymap[y, x]])

    if nFeature != -1:
        kps.sort(key=getScore)
        kps = kps[:nFeature]
        print kps.__len__(), 'keypoints were selected.'
    return kps


def drawKeypoints(img, kps):
    for kp in kps:
        pt = (kp[0], kp[1])
        cv2.circle(img, pt, 3, [0, 0, 255], 1, cv2.LINE_AA)
    return img


def getHarrisKps(img_path, win_size=3, nonMax_size=3, nonMaxValue=0, nFeature=-1, thCornerMap=-1, sigma=0.8):
    print "reading images..."
    img_rgb = cv2.imread(img_path)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_w = img.shape[1]
    img_h = img.shape[0]

    corner_map = np.zeros(img.shape, np.float)

    print "calculating corner response value..."
    _, gauss = getGaussianFilter(win_size=win_size, sigma=sigma)

    safe_range = 1 + win_size
    for j in range(safe_range, img_h - safe_range):
        for i in range(safe_range, img_w - safe_range):
            corner_map[j, i] = calcCornerResponse(img, i, j, gauss)

    print "thresholding corner map..."
    th_corner = thresholdCornerMap(corner_map, th=thCornerMap)

    print "non-maximum supression for corner map..."
    for i in range(safe_range, img_h - safe_range):
        for j in range(safe_range, img_w - safe_range):
            win, stx, enx, sty, eny = getWindowWithRange(th_corner, i, j, nonMax_size)
            nonMax_win, row, col = nonMaximumSupression(win)
            th_corner[stx:enx, sty:eny] = nonMax_win

    print "getting keypoints..."
    kps = getKeypoints(th_corner, nonMaxValue=nonMaxValue, nFeature=nFeature)
    return kps


# win_size=5, sigma=1.4
# win_size=5, sigma=1
# win_size=7, sigma=0.84089642
# win_size=3, sigma=0.8


if __name__ == '__main__':
    kps = getHarrisKps("img.jpg")
    img_kps = drawKeypoints(cv2.imread("img.jpg"), kps)
    cv2.imwrite("res.jpg", img_kps)
