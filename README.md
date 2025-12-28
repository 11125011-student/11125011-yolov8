---

# 實戰 | 使用YOLOv8實現面部表情識別

此報告使用 Ultralytics **YOLOv8** 進行「人臉表情」偵測：模型輸出每張臉的 bounding box，並同時分類表情（如 angry/happy/sad/neutral…）。

---

## 一、動機與目標

一般的表情辨識常以「分類」方式處理：輸入一張裁切好的臉 → 輸出表情。  
但在真實場景（照片、監視器、課堂鏡頭）往往同時出現多張人臉，因此我們希望系統能做到：
- 1. 在一張影像中 **自動找出所有人臉位置**（定位）
- 2. 對每張人臉 **判斷其表情類別**（辨識）

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

### 3:安裝 YOLOv8（Ultralytics）
Ultralytics 提供一整套流程（train/val/predict/export），你才能用最標準的方式完成作業並寫得有依據。
```python
!pip install ultralytics
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_3.png?raw=true" />

### 4-1.預先建立資料夾，以免解壓後找不到路徑
- YOLO 訓練需要固定的資料夾結構（images/labels + train/val 或 valid）
- 路徑固定後，`data.yaml` 才不會一直找不到（這是新手最常卡的點）
```python
!mkdir -p /content/datasets
!unzip -q "/content/drive/MyDrive/AI_114/Emotions.zip" -d /content/datasets
!ls -lah /content/datasets | head
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_4-1.png?raw=true" />

### 4-2.確認解壓後結構
```python
!ls /content/
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_4-2.png?raw=true" />

### 4-3.檢查路徑與類別
Ultralytics 會靠 `YAML` 讀到：
- train/val（或 valid）路徑
- 類別數 `nc`
- 類別名稱 `names`
沒有或寫錯就會直接 train 失敗。
```python
!cat /content/datasets/data.yaml
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_4-3.png?raw=true" />

### 5-1.開始訓練
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

### 5-2.確認訓練輸出資料夾
```python
!ls -lah "/content/drive/MyDrive/yolo_project"
```
```python
EXP_DIR = "/content/drive/MyDrive/yolo_project/exp14"  # <<< 如果不是 exp1，改成你實際那個資料夾
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_5-2.png?raw=true" />

### 5-3.檢查 weights 是否存在
```python
!ls -lah "/content/drive/MyDrive/yolo_project/exp14/weights"
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_5-3.png?raw=true" />

### 6-1.觀看訓練結果圖
```python
!ls -lah "/content/drive/MyDrive/yolo_project/exp14" | grep -E "\.png|\.jpg" || true
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_6-1.png?raw=true" />

### 6-2.顯示圖片
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

### 7-1.用 best.pt 做推論（predict），先用 valid/images
```python
from ultralytics import YOLO

best = f"{EXP_DIR}/weights/best.pt"
model = YOLO(best)

SOURCE = "/content/datasets/valid/images"
model.predict(source=SOURCE, imgsz=640, conf=0.5, max_det=100, save=True)
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_7-1.png?raw=true" />

### 7-2.找推論輸出資料夾
期末展示一定要有「實際預測框框 + 表情類別」圖片。Predict mode 也是官方標準流程。
```python
!ls -lt /content/runs/detect | head -n 20
```
```python
!ls -lah /content/runs/detect/predict | head
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_7-2.png?raw=true" />

### 8.把推論結果存回 Drive
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

### 9.(可選)匯出模型（ONNX）
**ONNX** 方便未來部署到不同平台/推論框架
```python
!yolo export model="/content/drive/MyDrive/yolo_project/exp1/weights/best.pt" format=onnx
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_9.png?raw=true" />

---

## 四、評估方式與結果呈現建議

Ultralytics 在訓練與驗證階段通常會輸出以下指標（可直接引用 runs/exp*/results.png）：

- mAP（mAP50 / mAP50-95）：衡量偵測整體準確度
- Precision：誤報（False Positive）控制能力
- Recall：漏報（False Negative）控制能力

另外，混淆矩陣可協助分析「哪些表情容易混淆」，常見案例：
- neutral 與 content（或輕微微笑）邊界模糊
- surprise 與 fear 在某些表情上相似

> 建議在 GitHub `assets/` 放 3 張核心圖：results.png、confusion_matrix.png、demo_pred.jpg。

---

## 五、討論：本方法的優勢與限制

### 優勢
- 一個模型同時做到 **多臉定位 + 表情分類**
- 流程完整：train → predict → export，容易重現與展示

### 限制
- 表情本身是「細粒度差異」，偵測框內的微表情可能需要更高解析度或更精細資料
- 類別可能不平衡（某些表情較少），導致模型偏向常見類別
- 只用 yolov8n + epochs=10 主要是完成作業展示；若追求更高精度需更多訓練與調參

---

### 6-1.觀看訓練結果圖
```python
!ls -lah "/content/drive/MyDrive/yolo_project/exp14" | grep -E "\.png|\.jpg" || true
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_6-1.png?raw=true" />

### 6-1.觀看訓練結果圖
```python
!ls -lah "/content/drive/MyDrive/yolo_project/exp14" | grep -E "\.png|\.jpg" || true
```
<img width="935" height="268" alt="s1" src="https://github.com/11125011-student/11125011-yolov8/blob/main/yolo_v8_6-1.png?raw=true" />





---

## 步驟 2：獲取並載入資料集

我們使用內建的 **Olivetti Faces** 資料集。這個資料庫包含 40 個人、每人 10 張（共 400 張）64x64 像素的灰階照片。系統會自動下載並將圖像像素歸一化處理。

```python
print("正在載入 Olivetti Faces 資料集...")
faces = fetch_olivetti_faces()

# X 為特徵數據 (400張圖像，每張已被攤平成 4096 個像素點)
# y 為目標標籤 (代表 40 個不同的人，編號 0-39)
X = faces.data
y = faces.target

print(f"資料載入完成！共有 {X.shape[0]} 張圖像，每張圖像特徵數：{X.shape[1]}")

```
<img width="1283" height="575" alt="s2" src="https://github.com/user-attachments/assets/334a8e3a-2342-46e0-988f-99b78c72b61f" />

---

## 步驟 3：分割訓練集與測試集

為了驗證模型的好壞，我們將資料分為兩部分：**訓練集**（讓機器學習）與**測試集**（檢查機器學得好不好）。這裡我們採用 8:2 的比例進行分割。

```python
# 使用 stratify=y 確保訓練集和測試集中每一類人的比例均勻
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"訓練樣本數: {len(X_train)}")
print(f"測試樣本數: {len(X_test)}")

```
<img width="841" height="143" alt="s3" src="https://github.com/user-attachments/assets/9ae26eea-9038-4525-83f4-bbbbbb9dff03" />

---

## 步驟 4：建立人工神經網路 (ANN) 模型

我們使用 `MLPClassifier` (多層感知器)。這是一種經典的人工神經網路。我們設定 200 個隱藏層神經元，並使用 `logistic` 激活函數。

```python
n_neurons = 200  # 設定隱藏層神經元數量

model = MLPClassifier(
    hidden_layer_sizes=(n_neurons,), 
    solver='adam',           # 優化演算法
    activation='logistic',   # 激活函數
    batch_size=1,            # 每次處理一個樣本
    early_stopping=True,     # 當驗證分數不再提升時提早停止，避免過擬合
    random_state=42,
    max_iter=500             # 最大迭代次數
)

```
<img width="886" height="375" alt="s4" src="https://github.com/user-attachments/assets/e7591b41-0da1-4269-a0b3-ddad732167fe" />

---

## 步驟 5：訓練模型

將訓練集的數據餵入模型。此時神經網路會不斷調整內部的權重，以試圖準確地將圖像像素與正確的身份標籤對應起來。

```python
print("模型訓練開始，請稍候...")
model.fit(X_train, y_train)
print("模型訓練完成！")

```
<img width="459" height="216" alt="s5" src="https://github.com/user-attachments/assets/13a7b590-a99a-44cb-8580-27a027633ebb" />

---

## 步驟 6：預測與效能評估

我們讓模型嘗試辨識它從未看過的測試集圖像，並輸出詳細的分類報告（包括準確度、召回率等指標）。

```python
# 進行預測
y_pred = model.predict(X_test)

# 顯示分類報告
print("\n--- 識別效能報告 ---")
print(classification_report(y_test, y_pred))

```
<img width="742" height="614" alt="s6" src="https://github.com/user-attachments/assets/ebf47d7f-dc24-4d00-a6df-7c5771327022" />

---

## 步驟 7：可視化混淆矩陣

混淆矩陣可以讓我們直觀地看到哪些類別被正確識別，以及模型最容易把誰認錯（矩陣對角線越深顏色，代表準確度越高）。

```python
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=False, cmap='Greens')
plt.title('Confusion Matrix - Olivetti Faces')
plt.xlabel('Predicted Identity')
plt.ylabel('True Identity')
plt.show()

```
<img width="1273" height="710" alt="s7" src="https://github.com/user-attachments/assets/ea1e091f-f493-4cbd-9530-f24aef3e2b4e" />

---

## 步驟 8：隨機測試結果展示

最後，我們隨機從測試集中選出 5 張照片，並將模型的「預測結果 (Pred)」與「真實答案 (True)」標註在圖片上方。

```python
plt.figure(figsize=(15, 5))
indices = np.random.choice(len(X_test), 5, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[idx].reshape(64, 64), cmap='gray')
    plt.title(f"True ID: {y_test[idx]}\nPred ID: {y_pred[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

```
<img width="1359" height="595" alt="s8" src="https://github.com/user-attachments/assets/ef0360c1-cd2a-4216-9c0a-2531c23ab77e" />


---

### **實作心得**

在這次的機器學習實作中，我們選擇了 **Olivetti Faces** 作為開發的核心資料庫。雖然這是一個規模相對輕量、且預處理完善的資料集，但對我們而言，這正是一個極佳的切入點，讓我們能拋開繁雜的資料清洗過程，將核心精力專注於 **人工神經網路 (ANN)** 的底層建構與運算邏輯。

#### **從理論到實踐：建構 ANN 模型**

透過這次報告，我們深刻體會到建立一個 ANN 模型並非單純地撰寫程式碼，而是一場關於「參數設計」與「邏輯推演」的過程。在定義 `MLPClassifier` 的過程中，我們學習到了：

* **隱藏層的配置**：如何決定神經元的數量以平衡運算效能與識別準確度。
* **激活函數的影響**：理解 `logistic` 或 `relu` 等函數如何幫助神經網路處理非線性的人臉特徵。
* **權重優化的關鍵**：學習到 `adam` 等優化器如何在訓練過程中，透過反向傳播不斷修正誤差，最終讓模型從一堆模糊的像素點中辨識出正確的身分。

#### **訓練過程中的挑戰與體會**

在訓練模型的階段，看著模型從最初的低準確率，隨著訓練輪數（Epochs）增加而逐漸收斂，這種從數據中發現模式的過程讓我非常有成就感。尤其是在分析「混淆矩陣」時，我們觀察到機器是如何區分細微的臉部差異，這讓我們對機器學習中的「特徵提取」有了實質的認知，而不僅僅停留在課本上的名詞解釋。

#### **受益良多的學習成果**

這次的實作讓我受益匪淺，最大的收穫在於我們跨越了「從無到有」的門檻。在 **Google Colab** 環境下，利用輕量級資料庫進行快速迭代與驗證，讓我們建立了一套完整的開發工作流：從資料載入、訓練集分割、模型調優到最後的視覺化驗證。這不僅鍛鍊了我們的 Python 實作能力，更為我們未來挑戰更大規模的影像處理（如 CNN 卷積神經網路）奠定了紮實的基礎。

---
