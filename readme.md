

# contrastive-axis-projection
## *A detector for routing philosophical contexts*

---

## data structure

- 100_base_d8_bge.npy       ← 由bge-large做的8軸axis做成的哲學方向矩陣
- build_100_base_d8.py      ← 生成d8base
- 100_meta.json             ← 資料說明集
- 100_axis.json             ← 8軸變異軸參考點
- design.md                 ← 設計內容
- readme.md                 ← 使用說明

---

## 100_axis.json 
### 設計層

哲學對立設計
與模型無關
可版本化
可擴充

---

## 100_base_d8_bge.npy
### 向量層

由 build_100_base_d8.py

讀 100_axis.json 逐軸取 A.sentences、B.sentences

用 bge-large-zh-v1.5 做 embedding，逐向量 normalize

計算 axis_raw = mean(B) - mean(A) 並 normalize，組成 AxisRaw

用 QR 正交化：Q, R = np.linalg.qr(AxisRaw.T)，AxisOrtho = Q.T

存成 100_base_d8_bge.npy