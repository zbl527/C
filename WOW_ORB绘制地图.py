# %%
import win32gui
import win32ui
import win32con
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time
import pyautogui

# 获取当前鼠标的位置
def get_mouse_position():
    return pyautogui.position()

# 获取窗口区域截图
def capture_window_region(hwnd, x, y, width, height):
    window_dc = win32gui.GetWindowDC(hwnd)
    dc_obj = win32ui.CreateDCFromHandle(window_dc)
    compatible_dc = dc_obj.CreateCompatibleDC()

    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(dc_obj, width, height)
    compatible_dc.SelectObject(bitmap)
    compatible_dc.BitBlt((0, 0), (width, height), dc_obj, (x, y), win32con.SRCCOPY)

    bitmap_info = bitmap.GetInfo()
    bitmap_bits = bitmap.GetBitmapBits(True)

    img = Image.frombuffer(
        'RGB',
        (bitmap_info['bmWidth'], bitmap_info['bmHeight']),
        bitmap_bits, 'raw', 'BGRX', 0, 1
    )

    # Cleanup
    win32gui.DeleteObject(bitmap.GetHandle())
    compatible_dc.DeleteDC()
    dc_obj.DeleteDC()
    win32gui.ReleaseDC(hwnd, window_dc)

    return img

# 根据窗口标题查找窗口
def find_window_by_title(title):
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        raise Exception(f"Window with title '{title}' not found!")
    return hwnd

# 应用中心遮罩
def apply_center_mask(image, mask_size):
    width, height = image.size
    cx, cy = width // 2, height // 2
    half_size = mask_size // 2

    draw = ImageDraw.Draw(image)
    draw.rectangle([cx - half_size, cy - half_size, cx + half_size, cy + half_size], fill=(0, 0, 0))

    return image

# 预处理图像
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(equalized_rgb)

# 获取多尺度特征
def get_multiscale_features(image, mask_size, scales=[1.0, 0.8, 0.6, 0.4, 0.2]):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    cx, cy = width // 2, height // 2
    half_size = mask_size // 2

    mask = np.ones((height, width), dtype=np.uint8) * 255
    mask[cy - half_size:cy + half_size, cx - half_size:cx + half_size] = 0

    all_keypoints = []
    all_descriptors = []

    for scale in scales:
        scaled_img = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
        scaled_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
        orb = cv2.ORB_create(nfeatures=10000)
        keypoints, descriptors = orb.detectAndCompute(scaled_img, scaled_mask)
        if descriptors is not None:
            all_keypoints.extend(keypoints)
            if len(all_descriptors) == 0:
                all_descriptors = descriptors
            else:
                all_descriptors = np.vstack((all_descriptors, descriptors))

    return all_keypoints, np.array(all_descriptors)

# 匹配特征点
def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# 计算移动向量
def calculate_movement(kp1, kp2, matches, distance_threshold):
    movements = []
    for match in matches:
        if match.distance < distance_threshold:
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt)
            movement = pt2 - pt1
            movements.append(movement)
    if len(movements) > 0:
        average_movement = np.median(movements, axis=0)
        return average_movement, True
    else:
        return np.array([0, 0]), False

# 使窗口保持在最上层
def keep_window_on_top(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

# 平滑移动向量
def smooth_movement(movements, alpha=0.5):
    if len(movements) < 2:
        return movements[-1]
    return alpha * movements[-1] + (1 - alpha) * movements[-2]

# 计算误差并调整位置
def correct_position(current_position, new_position, threshold=10):
    if np.linalg.norm(new_position - current_position) > threshold:
        return current_position
    return new_position

# 地图匹配校正位置
def match_and_correct_position(map_image, screenshot, current_position, kp_map, des_map, distance_threshold=35):
    kp_screenshot, des_screenshot = get_multiscale_features(screenshot, mask_size)
    matches = match_features(des_map, des_screenshot)
    movement, reliable = calculate_movement(kp_map, kp_screenshot, matches, distance_threshold)
    if reliable:
        new_position = current_position + movement
        new_position = correct_position(current_position, new_position)
        return new_position, kp_screenshot, des_screenshot, True
    return current_position, kp_screenshot, des_screenshot, False

# 更新位置并拼接图像
def update_position_and_stitch_image(movement_vector, screenshot, reliable, edge_thickness=1):
    global current_position, map_image, previous_movements, drawn_areas
    if reliable:
        movement_vector *= movement_scale
        previous_movements.append(movement_vector)
        smoothed_movement = smooth_movement(previous_movements)

        new_position = current_position - smoothed_movement
        new_position = correct_position(current_position, new_position)

        current_position_int = tuple(current_position.astype(int))
        new_position_int = tuple(new_position.astype(int))

        if (0 <= new_position_int[0] < map_size[0] and 
            0 <= new_position_int[1] < map_size[1]):
            
            cv2.line(map_image, current_position_int, new_position_int, path_color, path_thickness)

            screenshot_array = np.array(screenshot)
            screenshot_height, screenshot_width = screenshot_array.shape[:2]
            top_left_x = new_position_int[0] - screenshot_width // 2
            top_left_y = new_position_int[1] - screenshot_height // 2
            bottom_right_x = top_left_x + screenshot_width
            bottom_right_y = top_left_y + screenshot_height

            if (0 <= top_left_x < map_size[0] and 0 <= top_left_y < map_size[1] and
                0 <= bottom_right_x < map_size[0] and 0 <= bottom_right_y < map_size[1]):
                
                for i in range(top_left_y, bottom_right_y):
                    for j in range(top_left_x, bottom_right_x):
                        # Check if the pixel is within the edge thickness range
                        if (drawn_areas[i, j] == 0 or
                            any(abs(i - edge_y) < edge_thickness for edge_y in [top_left_y, bottom_right_y - 1]) or
                            any(abs(j - edge_x) < edge_thickness for edge_x in [top_left_x, bottom_right_x - 1])):
                            map_image[i, j] = screenshot_array[i - top_left_y, j - top_left_x]

                drawn_areas[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 1

            current_position = new_position

# 初始化参数
window_title = '魔兽世界'
x, y, width, height = 1765, 93, 95, 95 # 根据实际窗口大小调整
mask_size = 23 # 中心掩码大小
distance_threshold = 35 # 距离阈值

# 初始化地图参数
map_size = (1000, 1000) # 地图大小，根据实际需求调整
map_center = (map_size[0] // 2, map_size[1] // 2)
current_position = np.array(map_center)
path_color = (0, 255, 0)
path_thickness = 2  # 速度缩放因子
movement_scale = 1 # 速度缩放因子

map_image = np.ones((map_size[0], map_size[1], 3), dtype=np.uint8) * 255
drawn_areas = np.zeros(map_size, dtype=np.uint8)

previous_movements = []
update_interval = 0.5 # 更新间隔，单位为秒
edge_thickness = 3 # 边缘厚度，用于绘制路径时避免边缘像素被覆盖
# 地图特征点和描述符
kp_map = []
des_map = []

try:
    hwnd = find_window_by_title(window_title)
    prev_screenshot = capture_window_region(hwnd, x, y, width, height)
    prev_screenshot = apply_center_mask(prev_screenshot, mask_size)
    prev_screenshot = preprocess_image(prev_screenshot)
    prev_kp, prev_des = get_multiscale_features(prev_screenshot, mask_size)

    cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
    
    while True:
        start_time = time.time()
        
        screenshot = capture_window_region(hwnd, x, y, width, height)
        screenshot = apply_center_mask(screenshot, mask_size)
        screenshot = preprocess_image(screenshot)
        kp, des = get_multiscale_features(screenshot, mask_size)
        matches = match_features(prev_des, des)
        movement, reliable = calculate_movement(prev_kp, kp, matches, distance_threshold)

        update_position_and_stitch_image(movement, screenshot, reliable, edge_thickness)

        # 地图匹配校正位置
        new_position, kp_map, des_map, reliable_map = match_and_correct_position(map_image, screenshot, current_position, kp, des, distance_threshold)
        if reliable_map:
            current_position = new_position

        prev_kp, prev_des = kp, des

        cv2.imshow('Map', map_image)
        keep_window_on_top('Map')
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            mouse_x, mouse_y = get_mouse_position()
            x, y = mouse_x - width // 2, mouse_y - height // 2
        
        elapsed_time = time.time() - start_time
        if elapsed_time < update_interval:
            time.sleep(update_interval - elapsed_time)

    cv2.imwrite('final_map.png', map_image)
    cv2.destroyAllWindows()
except Exception as e:
    print(f"发生错误: {e}")



