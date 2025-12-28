---

# 實戰 | 使用YOLOv8實現面部表情識別

此報告使用 Ultralytics **YOLOv8** 進行「人臉表情」偵測：模型輸出每張臉的 bounding box，並同時分類表情（如 angry/happy/sad/neutral…）。

---

## 一、動機與目標

一般的表情辨識常以「分類」方式處理：輸入一張裁切好的臉 → 輸出表情。  
但在真實場景（照片、監視器、課堂鏡頭）往往同時出現多張人臉，因此我們希望系統能做到：
- 在一張影像中 **自動找出所有人臉位置**（定位）
- 對每張人臉 **判斷其表情類別**（辨識）

---

## 二、工具

- 平台：Google Colab
- 深度學習架構：Ultralytics YOLOv8
- 資料集來源：<src="https://universe.roboflow.com/emotions-dectection/human-face-emotions/dataset/30/download/yolov8" />
- 資料參考:<src="https://zhuanlan.zhihu.com/p/1927118824295102227" />

---

## 三、實際步驟與流程 

### 1:開 Google Colab + 開 GPU
Colab：執行階段 → 變更執行階段類型 → 硬體加速器選 GPU
YOLOv8 訓練是大量矩陣運算，需要顧及訓練速度
<img width="900" height="900" alt="s8" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_1.png" />

### 2:掛載 Google Drive
Colab 的 `content` 是暫存，重開就清掉；Drive 才能保存：
- 資料集 `zip`
- 訓練產出的 `best.pt`、曲線圖、混淆矩陣
- 推論結果圖片（方便放到 GitHub）
```python
from google.colab import drive
drive.mount('/content/drive')
```
<img width="900" height="600" alt="s8" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_2.png?raw=true" />

### 3 : 安裝 YOLOv8（Ultralytics）
Ultralytics 提供一整套流程（train/val/predict/export），你才能用最標準的方式完成作業並寫得有依據。
```python
!pip install ultralytics
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_3.png?raw=true" />

### 4-1:預先建立資料夾，以免解壓後找不到路徑
- YOLO 訓練需要固定的資料夾結構（images/labels + train/val 或 valid）
- 路徑固定後，`data.yaml` 才不會一直找不到（這是新手最常卡的點）
```python
!mkdir -p /content/datasets
!unzip -q "/content/drive/MyDrive/AI_114/Emotions.zip" -d /content/datasets
!ls -lah /content/datasets | head
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_4-1.png?raw=true" />

### 4-2:確認解壓後結構
```python
!ls /content/
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_4-2.png?raw=true" />

### 4-3:檢查路徑與類別
Ultralytics 會靠 `YAML` 讀到：
- train/val（或 valid）路徑
- 類別數 `nc`
- 類別名稱 `names`
沒有或寫錯就會直接 train 失敗。
```python
!cat /content/datasets/data.yaml
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_4-3.png?raw=true" />

### 5-1:開始訓練
- 用 yolov8n：在 Colab 時間/資源有限時，先跑通流程、先有成果，再談調參。
- epochs=10：作業展示用；若要提升精度再加 epochs。
- imgsz=640：偵測常用解析度，太小可能抓不到臉部細節，太大訓練慢。
- batch=8：配合 GPU 顯存，避免 OOM。
```python
!yolo task=detect \
  mode=train \
  model=yolov8n.pt \
  data=/content/datasets/data.yaml \
  epochs=10 \
  imgsz=640 \
  batch=8 \
  project=/content/drive/MyDrive/yolo_project \
  name=exp1
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_5-1.png?raw=true" />

### 5-2:確認訓練輸出資料夾
```python
!ls -lah "/content/drive/MyDrive/yolo_project"
```
```python
EXP_DIR = "/content/drive/MyDrive/yolo_project/exp14"  # <<< 如果不是 exp1，改成你實際那個資料夾
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_5-2.png?raw=true" />

### 5-3:檢查 weights 是否存在
```python
!ls -lah "/content/drive/MyDrive/yolo_project/exp14/weights"
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_5-3.png?raw=true" />

### 6-1:觀看訓練結果圖
```python
!ls -lah "/content/drive/MyDrive/yolo_project/exp14" | grep -E "\.png|\.jpg" || true
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_6-1.png?raw=true" />

### 6-2:顯示圖片
報告最重要產物就是：
- `best.pt`（最佳權重）
- `results.png`（訓練曲線）
- `confusion_matrix.png`（類別混淆）
這些就是你報告的「實驗結果圖」。
```python
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
exp = Path(EXP_DIR)
files = [
    exp/"results.png",
    exp/"confusion_matrix.png",
    exp/"confusion_matrix_normalized.png",
    exp/"val_batch0_pred.jpg"
]
for f in files:
    if f.exists():
        img = Image.open(f)
        plt.figure(figsize=(10,6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f.name)
        plt.show()
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_6-2.png?raw=true" />
輸出「結果」:整個訓練過程的總覽記錄
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/result-1.png?raw=true" />
輸出「混淆矩陣」:哪些表情最容易被誤判成哪些表情
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/result-2.png?raw=true" />
輸出「歸一化混淆矩陣」（比例版）:更容易看出每一類自己的誤判結構
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/result-3.png?raw=true" />
輸出「驗證集預測結果」:拿來做報告展示、或肉眼檢查模型是不是在亂框/漏框，通常可視為「驗證集的隨機樣本」
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/result-4.png?raw=true" />

### 7-1:用 best.pt 做推論（predict），先用 valid/images
```python
from ultralytics import YOLO

best = f"{EXP_DIR}/weights/best.pt"
model = YOLO(best)

SOURCE = "/content/datasets/valid/images"
model.predict(source=SOURCE, imgsz=640, conf=0.5, max_det=100, save=True)
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_7-1.png?raw=true" />

### 7-2:找推論輸出資料夾
期末展示一定要有「實際預測框框 + 表情類別」圖片。Predict mode 也是官方標準流程。
```python
!ls -lt /content/runs/detect | head -n 20
```
```python
!ls -lah /content/runs/detect/predict | head
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_7-2.png?raw=true" />

### 8:把推論結果存回 Drive
`runs` 在 Colab 暫存，不搬回 Drive 就沒了；而 GitHub 只需要放幾張代表性成果圖，不用放整包資料集。
```python
import shutil
from datetime import datetime

src = "/content/runs/detect/predict"  # <<< 改成你最新的 predict 資料夾
dst = f"/content/drive/MyDrive/yolo_project/predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
shutil.copytree(src, dst, dirs_exist_ok=True)
print("Saved to:", dst)
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_8.png?raw=true" />

### 9:(可選)匯出模型（ONNX）
**ONNX** 方便未來部署到不同平台/推論框架
```python
!yolo export model="/content/drive/MyDrive/yolo_project/exp1/weights/best.pt" format=onnx
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_9.png?raw=true" />

---

## 四、評估方式與建議

Ultralytics 在訓練與驗證階段通常會輸出以下指標（可直接引用 runs/exp*/results.png）：

- mAP（mAP50 / mAP50-95）：衡量偵測整體準確度
- Precision：誤報（False Positive）控制能力
- Recall：漏報（False Negative）控制能力

另外，混淆矩陣可協助分析「哪些表情容易混淆」，常見案例：
- neutral 與 content（或輕微微笑）邊界模糊
- surprise 與 fear 在某些表情上相似

> 建議在 GitHub `assets/` 放 3 張核心圖：results.png、confusion_matrix.png、demo_pred.jpg。

---

## 五、優勢與限制

### 1:優勢
- 一個模型同時做到 **多臉定位 + 表情分類**
- 流程完整：train → predict → export，容易重現與展示

### 2:限制
- 表情本身是「細粒度差異」，偵測框內的微表情可能需要更高解析度或更精細資料
- 類別可能不平衡（某些表情較少），導致模型偏向常見類別
- 只用 yolov8n + epochs=10 主要是完成作業展示；若追求更高精度需更多訓練與調參

---

## 六、延伸

- 增加訓練 epochs、嘗試 yolov8s/m
- 針對類別不平衡：加權、重採樣、蒐集更多少數類別資料
- 兩階段架構：先用人臉偵測器（或人臉對齊）→ 再做表情分類（通常更穩）
- 若要影片即時：調整 conf、max_det，並評估 FPS 與延遲

--- 

## 七、實作心得

這次以 YOLOv8 進行人臉表情辨識的實作，讓我第一次把「表情分類」從單一人臉的影像分類，提升到更貼近真實情境的「多臉偵測＋表情判別」。

在 Colab 上從資料集解壓、檢查 data.yaml、設定 epochs/imgsz/batch，到完成訓練與推論，我更清楚 YOLO 流程其實是一套可重現的工程：資料結構決定能否訓練、超參數影響速度與精度，而輸出檔（results、confusion_matrix、val_batch0_pred）則是理解模型行為的關鍵。results 曲線讓我看到 loss 下降與 mAP 變化的關係；混淆矩陣則直接揭露哪些表情最容易互相誤判（例如 neutral 與輕微微笑類），提醒我「準確率」之外更要看錯在哪裡。從 val_batch0_pred 的視覺化，我也發現即便框抓得到臉，分類仍可能受角度、遮擋、光線與表情細微差異影響。

我們認為YOLOv8 的優勢是速度快、端到端易部署，但限制也很明顯：資料不平衡與標註一致性會放大錯誤。未來我會嘗試增加訓練輪數、改用較大模型、做類別平衡與更精細的增強，甚至採兩階段（先人臉偵測再表情分類）來提升穩定性。

