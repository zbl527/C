# %%
import os
import time
import pytesseract
from PIL import Image,ImageEnhance
import tkinter as tk
import tkinter.font as tkFont
import math
import win32gui
import win32ui
import win32con
import random
# 获取当前目录并设置Tesseract路径
current_directory = os.getcwd()
tesseract_path = os.path.join(current_directory, "Tesseract-OCR", "tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# %%
# 定义键值
hit_key = 0x30 # 0键
clear_key = 0x39 # 9键
tab_key = 0x09 # TAB键
run_key = 0x57 # W键
jump_key = 0x20 # 空格键
left_turn_key = 0x51  # Q键
right_turn_key = 0x45  # E键


# %%
last_ahp = None  # 上一次检查时的 ahp 值
ahp_unchanged_count = 0  # ahp 值未改变的次数
max_unchanged_count = 3  # ahp 值未改变的最大次数，超过这个值将切换到非战斗状态
no_change_counter = 0
pve = 0
targets = [ 
{'x': 38.1, 'y': 21.1},
{'x': 35.3, 'y': 25.9},
{'x': 41.4, 'y': 33.9},
{'x': 47.1, 'y': 47.1},
{'x': 46.6, 'y': 54.5},
{'x': 54.3, 'y': 52.5},
{'x': 54.3, 'y': 50.8},
]
target_index = 0
prev_x, prev_y = None, None
delta_x, delta_y = 0, 0
hwnd = None
ranges = None
ahp = None
game_window = '魔兽世界'
def find_window(title):
    hwnd = win32gui.FindWindow(None, title)
    return hwnd
# 使用窗口的标题来查找窗口句柄
hwnd = find_window(game_window)

# %%
# 初始化Tkinter窗口和Canvas...
root = tk.Tk()
root.title("WOW!")
root.attributes("-topmost", True)
fontStyle = tkFont.Font(family="Lucida Grande", size=14)

canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack()
canvas.config(scrollregion=(0, 0, 400, 400))
x_data=[]
y_data=[]
x_coords = [0]
y_coords = [400]  # 由于Canvas的y坐标是从上往下增加的，所以用400-y来转换坐标系

labels = {}
for label in ['message', 'coords', 'php', 'pmg', 'ahp', 'ranges', 'combat']:
    labels[label] = tk.Label(root, text="", font=fontStyle)
    labels[label].pack()

def update_label(label, message):
    labels[label].config(text=message)
    root.update()

# %%

def capture_window(hwnd, capture_ratio=(74, 0.1, 17, 1.7)):
    # 获取窗口DC
    window_dc = win32gui.GetWindowDC(hwnd)
    dc = win32ui.CreateDCFromHandle(window_dc)
    compatible_dc = dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    
    # 获取窗口的完整尺寸
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bottom - top

    # 计算截图区域的尺寸和起始坐标
    capture_x = int(w * capture_ratio[0] / 100)
    capture_y = int(h * capture_ratio[1] / 100)
    capture_w = int(w * capture_ratio[2] / 100)
    capture_h = int(h * capture_ratio[3] / 100)

    # 创建位图对象 - 使用截图区域的尺寸
    bitmap.CreateCompatibleBitmap(dc, capture_w, capture_h)

    # 执行截图操作
    compatible_dc.SelectObject(bitmap)
    compatible_dc.BitBlt((0, 0), (capture_w, capture_h), dc, (capture_x, capture_y), win32con.SRCCOPY)

    # 转换为PIL图像格式
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)
    image = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)
    
    # 清理资源
    win32gui.DeleteObject(bitmap.GetHandle())
    compatible_dc.DeleteDC()
    dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, window_dc)

    return image


def convert_value(value):
    """Converts the value if greater than 100 by keeping only the first two digits and adding a decimal point before the third."""
    if value > 100:
        value_str = str(value)
        new_value = float(value_str[:2] + '.' + value_str[2:3])
    else:
        new_value = value
    return new_value

def add_data(x_data, y_data, current_x, current_y):
    """Adds the current data point to the lists if it's within the expected range compared to the average of the last 5 points."""
    current_x = convert_value(current_x)
    current_y = convert_value(current_y)

    if len(x_data) >= 5 and len(y_data) >= 5:
        # Calculate the average of the last 5 data points
        avg_x = sum(x_data[-5:]) / 5
        avg_y = sum(y_data[-5:]) / 5

        # Check if the new data points are within the acceptable range
        if abs(current_x - avg_x) > 3 or abs(current_y - avg_y) > 3:
            print("偏差太大，忽略此数据。")
            return x_data[-1], y_data[-1]

    x_data.append(current_x)
    y_data.append(current_y)
    return current_x, current_y


# %%
def get_window_coordinates():
    screenshot = capture_window(hwnd)
    # 图像处理：灰度化、对比度增强
    gray_image = screenshot.convert('L')
    enhancer = ImageEnhance.Contrast(gray_image)
    contrasted_image = enhancer.enhance(2)



    # 使用 Tesseract OCR 识别文本
    text = pytesseract.image_to_string(contrasted_image, lang='eng', config='--psm 6 -c tessedit_char_whitelist=0123456789.,')
    
    # 解析提取的文本信息
    info = text.replace("\n", "").replace('...', '.').replace('..', '.').split(',')

    def safe_float_convert(value, default=0.0):
        try:
            return float(value)
        except ValueError:
            return default
    def safe_int_convert(value, default=0):
        try:
            return int(value)
        except ValueError:
            return default
    # 使用安全转换函数确保除坐标外的所有值都能被转换为整数
    php, pmg, ahp, ranges, combat = [safe_int_convert(value) for value in info[2:7]]

    # 直接将坐标转换为浮点数
    current_x, current_y = safe_float_convert(info[0]), safe_float_convert(info[1])
    
    # 确保坐标在合理范围内
    current_x = convert_value(current_x)
    current_y = convert_value(current_y)

    current_x,current_y = add_data(x_data, y_data,current_x, current_y)

    php = round(php * 0.1, 1) if php > 100 else php
    pmg = round(pmg * 0.1, 1) if pmg > 100 else pmg
    ahp = round(ahp * 0.1, 1) if ahp > 100 else ahp
    label_values = {
        'coords': f"当前坐标:{current_x},{current_y}",
        'php': f"玩家血量:{php}%",
        'pmg': f"玩家能量:{pmg}%",
        'ahp': f"目标血量:{ahp}%",
        'ranges': f"目标距离:{ranges}",
        'combat': f"状态:{'战斗' if combat > 0 else '空闲'}"
    }
    # 更新标签显示
    for label, value in label_values.items():
        update_label(label, value)
    
    return current_x,current_y,php,pmg,ahp,ranges,combat
# %%

def draw_target_point(x, y):
    # 转换坐标系，使得(0,0)位于左下角
    scale_factor = 400 / 100  # 假设Canvas是400x400，且目标坐标系是100x100
    canvas_x = x * scale_factor
    canvas_y = (100 - y) * scale_factor  # 100-y是因为Tkinter Canvas的y坐标是从上往下增长的

    # 绘制蓝色的目标点
    canvas.create_oval(canvas_x - 3, canvas_y - 3, canvas_x + 3, canvas_y + 3, fill="blue")

# 在初始化GUI和Canvas之后、开始主循环之前绘制所有目标点
for target in targets:
    draw_target_point(target['x'], target['y'])

def draw_point(x, y):
    # 转换坐标系，使得(0,0)位于左下角
    # 假设这些坐标是基于100x100的坐标系，需要缩放到Canvas的尺寸
    scale_factor = 400 / 100  # 假设Canvas是400x400，且目标坐标系是100x100
    canvas_x = x * scale_factor
    canvas_y = (100 - y) * scale_factor  # 100-y是因为Tkinter Canvas的y坐标是从上往下增长的

    # 绘制新的点并连接线，使用width参数设置线条宽度
    if len(x_coords) > 1:  # 确保有前一个点可以连接
        canvas.create_line(x_coords[-1], y_coords[-1], canvas_x, canvas_y, fill="red", width=2)  # width=2设置线条宽度为2
    else:  # 如果是第一个点，只绘制一个点
        canvas.create_oval(canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2, fill="red", width=2)  # 同样可以设置点的宽度

    # 更新坐标列表
    x_coords.append(canvas_x)
    y_coords.append(canvas_y)


# %%
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360

def calculate_time_to_sleep(angle_diff):
    if angle_diff > 180:
        angle_diff = 360 - angle_diff  # Adjust for reverse direction
    time_to_sleep = angle_diff / 180
    return time_to_sleep


def move_to_target(current_x, current_y, target_x, target_y, prev_x, prev_y,php,pmg,ahp,ranges,combat):
    global no_change_counter  # 使用 global 关键字来修改外部变量
    global last_ahp  # 添加这行来声明 last_ahp 是全局变量
    global ahp_unchanged_count  # 如果这也是全局变量，也需要声明
    no_change_threshold = 2
    current_angle = 0
    # 判断是否到达目的地（偏差范围内）
    if calculate_distance(current_x, current_y, target_x, target_y) <= 0.3:
        win32gui.PostMessage(hwnd, win32con.WM_KEYUP, run_key, 0)  # 停止前进
        update_label('message', f"到达目标")
        return None, None, True  # 到达目标

    print(current_x,current_y)
    if ranges and int(ranges) <= 30 and pmg >10:
        # 战斗状态
        current_ahp = int(ahp)  # 假设 ahp 是当前血量的变量

        # 检查 ahp 是否改变
        if current_ahp == last_ahp:
            ahp_unchanged_count += 1
        else:
            ahp_unchanged_count = 0

        last_ahp = current_ahp  # 更新上一次检查的 ahp 值

        # 检查 ahp 是否连续几次未改变
        if ahp_unchanged_count >= max_unchanged_count:
            # 随机选择左转或右转
            turn_key = random.choice([left_turn_key, right_turn_key])
            # 发送按下键的消息
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, turn_key, 0)
            time.sleep(1)  # 按键持续时间
            # 发送释放键的消息
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, turn_key, 0)
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, clear_key, 0)
            time.sleep(0.1) 
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, clear_key, 0)
            ahp_unchanged_count = 0
            pass
        else:
            # 战斗状态
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, run_key, 0)
            time.sleep(1)
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, hit_key, 0)
            time.sleep(0.1)
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, hit_key, 0)
    else:
        #非战斗状态
        update_label('message', f"正在前往:{target_x},{target_y}")
        # 计算坐标变化
        if prev_x is not None and prev_y is not None:
            delta_x = current_x - prev_x
            delta_y = current_y - prev_y

            # 如果坐标变化小，则增加无变化计数器
            if abs(delta_x) < 0.1 and abs(delta_y) < 0.1:
                no_change_counter += 1
            else:
                no_change_counter = 0

            # 如果超过无变化阈值，执行跳跃动作以解除可能的卡住状态
            if no_change_counter >= no_change_threshold:
                update_label('message', f"角色跳跃")
                win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, jump_key, 0)
                time.sleep(1)
                win32gui.PostMessage(hwnd, win32con.WM_KEYUP, jump_key, 0)
                no_change_counter = 0

            # 计算当前朝向
            current_angle = calculate_angle(prev_x, prev_y, current_x, current_y)

        # 计算目标方向
        target_angle = calculate_angle(current_x, current_y, target_x, target_y)

        # 计算角度差
        angle_diff = (target_angle - current_angle + 360) % 360

        # 计算根据角度差所需的时间
        time_to_sleep = calculate_time_to_sleep(angle_diff)
        # 确保time_to_sleep不超过1秒
        time_to_sleep = min(time_to_sleep, 0.5)
 
        # 判断转向和前进
        if angle_diff != 0:
            if angle_diff < 180:
                # 右转E
                win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, right_turn_key, 0)
                time.sleep(time_to_sleep)
                win32gui.PostMessage(hwnd, win32con.WM_KEYUP, right_turn_key, 0)
            else:
                # 左转Q
                win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, left_turn_key, 0)
                time.sleep(time_to_sleep)
                win32gui.PostMessage(hwnd, win32con.WM_KEYUP, left_turn_key, 0)
            # 向前移动W
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, run_key, 0)
            #按TAB切换目标
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, tab_key, 0)
            time.sleep(0.1)
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, tab_key, 0)
            ahp_unchanged_count = 0

    return current_x, current_y, False




# %%

def update_position():
    try:
        current_x, current_y, php, pmg, ahp, ranges, combat = get_window_coordinates()
    except Exception as e:
        print(e)  # 打印异常信息
        root.after(1000, update_position)  # 1秒后再次尝试
        return

    global prev_x, prev_y, target_index
    if target_index >= len(targets):
        update_label('message', "寻路完成")
    else:

        target_x, target_y = targets[target_index]['x'], targets[target_index]['y']
        def find_nearest_point_on_line(x0, y0, x1, y1, x2, y2):
            """
            计算点 (x0, y0) 到通过点 (x1, y1) 和 (x2, y2) 定义的线段的最近点
            """
            A = x0 - x1
            B = y0 - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D
            param = -1
            if len_sq != 0:  # 线段的长度不为0
                param = dot / len_sq

            if param < 0:
                xx = x1
                yy = y1
            elif param > 1:
                xx = x2
                yy = y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            dx = x0 - xx
            dy = y0 - yy
            return (xx, yy), math.sqrt(dx * dx + dy * dy)

        def update_target_point(current_x, current_y, path):
            """
            更新目标点以确保角色朝着路径移动
            """
            min_distance = float('inf')
            nearest_point = None

            # 遍历路径中的每个线段
            for i in range(len(path) - 1):
                point, distance = find_nearest_point_on_line(current_x, current_y, path[i][0], path[i][1], path[i+1][0], path[i+1][1])
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = point

            # 如果最短距离超过阈值，则更新目标点
            if min_distance > 2:
                return nearest_point
            else:
                # 如果当前点非常接近路径上的某个点，则直接使用该点作为目标点
                for target_point in path:
                    if calculate_distance(current_x, current_y, target_point[0], target_point[1]) <= 0.3:
                        return target_point
                return None

        # 使用更新后的目标点调整角色移动
        new_target_point = update_target_point(current_x, current_y, path)
        if new_target_point is not None:
            target_x, target_y = new_target_point


        
        prev_x, prev_y, reached = move_to_target(current_x, current_y, target_x, target_y, prev_x, prev_y, php, pmg, ahp, ranges, combat)
        draw_point(current_x, current_y)
        if reached:
            target_index += 1
    
    root.after(1500, update_position)  # 1秒后再次更新位置


root.after(1500, update_position)  # 启动位置更新循环
root.mainloop()
