#coding=utf-8
import cv2
import os
from glob import glob
import numpy as np

# 读取图像并进行车道线提取
def extract_lane_lines(image):
    # 图像预处理，例如灰度化、边缘检测等
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    ## 进行直线检测
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    ## 返回检测到的车道线
    #return lines
    # 过滤出在三角形区域内的直线
    triangle_points = np.array([[0, 800], [950, 550], [1919, 800]], np.int32)  # 根据您指定的三角形顶点坐标定义一个三角形

    # 创建掩膜图像，将三角形区域填充为白色，其余区域填充为黑色
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [triangle_points], 255)

    # 应用掩膜，只保留三角形区域内的边缘
    masked_edges = cv2.bitwise_and(edges, mask)

    # 进行直线检测
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
    return np.array(lines)

# 计算交点
def compute_vanishing_point(lines):
    # 将检测到的直线转换为二维数组
    lines = lines.squeeze()

    # 构建最小二乘法拟合所需的数据
    A = []
    b = []

    for Line in lines:
        x1, y1, x2, y2 = Line
        A.append([y1 - y2, x2 - x1])
        b.append([x2*y1 - x1*y2])

    # 使用最小二乘法求解消失点
    A = np.array(A)
    b = np.array(b)
    #x = np.linalg.lstsq(A, b, rcond=None)[0]
    #vanishing_point = [x[0], x[1]]  # 注意交换坐标位置

    # 计算A的奇异值分解
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # 计算伪逆矩阵
    S_inv = np.diag(1 / S)
    A_inv = np.dot(V.T, np.dot(S_inv, U.T))

    # 计算最小二乘解
    x = np.dot(A_inv, b)

    vanishing_point = [x[0, 0], x[1, 0]]  # 注意交换坐标位置


    # 返回消失点的坐标
    return vanishing_point

# 可视化车道线和消失点
def visualize_results(image, lane_lines, vanishing_point):
    # 在图像上绘制车道线
    for line in lane_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 在图像上绘制消失点
    if vanishing_point is not None:
        vx, vy = vanishing_point
        vx = int(vx)
        vy = int(vy)
        cv2.circle(image, (vx, vy), 5, (0, 255, 0), -1)

    # 显示图像
    cv2.imshow("Lane Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 处理所有图像
def process_images(image_files):
    for file in image_files:
        # 读取图像
        image = cv2.imread(file)

        # 提取车道线
        lane_lines = extract_lane_lines(image)

        # 计算消失点
        vanishing_point = compute_vanishing_point(lane_lines)

        # 可视化结果
        visualize_results(image, lane_lines, vanishing_point)

        # 打印消失点坐标
        print("Vanishing Point for", file, ":", vanishing_point)

# 图像文件列表
image_files = []
image_files += glob(os.path.join('./data', "*.jpeg"))
image_files += glob(os.path.join('./data', "*.jpg"))

# 处理图像
process_images(image_files)
