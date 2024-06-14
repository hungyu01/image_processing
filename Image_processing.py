import cv2
import numpy as np

# 定義各種效果的函數
def apply_effect(frame, effect):
    if effect == "gray":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif effect == "invert":
        return cv2.bitwise_not(frame)
    elif effect == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif effect == "canny":
        return cv2.Canny(frame, 100, 200)
    elif effect == "sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131], 
                                 [0.349, 0.686, 0.168], 
                                 [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        sepia_frame = np.clip(sepia_frame, 0, 255)  # 防止溢出
        return sepia_frame.astype(np.uint8)
    elif effect == "mosaic":
        size = frame.shape         # 取得原始圖片的資訊
        level = 15               # 縮小比例 ( 可當作馬賽克的等級 )
        h = int(size[0]/level)   # 按照比例縮小後的高度 ( 使用 int 去除小數點 )
        w = int(size[1]/level)   # 按照比例縮小後的寬度 ( 使用 int 去除小數點 )
        mosaic = cv2.resize(frame, (w,h), interpolation=cv2.INTER_LINEAR)   # 根據縮小尺寸縮小
        mosaic = cv2.resize(mosaic, (size[1],size[0]), interpolation=cv2.INTER_NEAREST) # 放大到原本的大小
        return mosaic.astype(np.uint8)
    elif effect == "negative1":
        return cv2.bitwise_not(frame)
    elif effect == "negative2":
        frame[:, :, 0] = cv2.bitwise_not(frame[:, :, 0])  # 仅对蓝色通道应用负片
        return frame
    elif effect == "negative3":
        frame[:, :, 1] = cv2.bitwise_not(frame[:, :, 1])  # 仅对绿色通道应用负片
        return frame
    return frame

# 讀取影片
cap = cv2.VideoCapture('video.mp4')

# 檢查影片是否正確打開
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# 檢查FPS是否有效
if fps == 0:
    print("Error: FPS value is 0. Check the video file.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

# 定義每10秒的效果列表
effects = ["original", "gray", "invert", "blur", "canny", "sepia", "mosaic", "negative"]
effect_interval = 15   # 每個效果的持續時間

# 設定影片寫入
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

current_effect_index = 0
current_effect = effects[current_effect_index]
effect_frames = int(effect_interval * fps)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % effect_frames == 0:
        current_effect_index = (current_effect_index + 1) % len(effects)
        current_effect = effects[current_effect_index]
    
    if current_effect == "negative":
        # 分割成三個畫面
        height, width = frame.shape[:2]
        third_width = width // 3
        part1 = frame[:, :third_width]
        part2 = frame[:, third_width:2 * third_width]
        part3 = frame[:, 2 * third_width:]

        # 對每個畫面應用不同的負片效果
        part1 = apply_effect(part1, "negative1")
        part2 = apply_effect(part2, "negative2")
        part3 = apply_effect(part3, "negative3")

        # 合併三個部分
        processed_frame = np.hstack((part1, part2, part3))
    else:
        processed_frame = apply_effect(frame, current_effect)
    
    # 如果處理後的幀是灰度圖像，將其轉換為BGR
    if len(processed_frame.shape) == 2:
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
    
    # 顯示當前的技術
    effect_text = f"Effect: {current_effect}"
    cv2.putText(processed_frame, effect_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 顯示即時的幀數/總幀數
    text = f"Frame: {frame_count + 1}/{total_frames}"
    cv2.putText(processed_frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 將原始影片作對比
    height, width = frame.shape[:2]
    thumbnail = cv2.resize(frame, (width // 4, height // 4))

    # 設置縮略圖的固定位置
    top_left_y = 90  # y座標
    top_left_x = 0   # x座標
    
    bottom_right_y = top_left_y + thumbnail.shape[0]
    bottom_right_x = top_left_x + thumbnail.shape[1]

    # 將縮略圖放置在指定位置
    processed_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = thumbnail

    out.write(processed_frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
