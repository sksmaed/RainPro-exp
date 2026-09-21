# RainPro-8-TW 實作細節紀錄

本文件記錄 RainPro-8-TW 程式碼層級的決策、推導與修正——「為什麼程式碼長這樣」，而不是實驗方法論。實驗臂設計、分階段評估計劃見 `docs/rainpro_tw_evaluation.md`；這裡只放會影響到怎麼讀/改程式碼的細節。

## `include_satellite` 與 8km tier 共用

`radar_8km`（QPESUMS）與 `satellite_8km`（STA_H8）共用 `tier="8km"`（`rainpro/data/rainpro8_sources.py`）。`include_satellite=False` 只從 `sources` dict 移除 `satellite_8km`，`radar_8km` 不受影響——這與 `include_gfs=False` 會讓整個 16km tier 消失、觸發 `rainpro.network.rainpro8.RainPro` 的 0-channel skip 分支不同（8km 分支本來就沒有這種 all-or-nothing 邏輯，channel 數變少即可，無需改網路架構）。已驗證：`build_taiwan_sources(include_satellite=False)` 的 8km tier channel 數 = `radar_8km` 單獨的 channel 數。

`rainpro/modules/rainpro8.py::stack_sources` 完全是動態依 `sources` dict 組 tensor（`[tensors[name] for name, spec in sources.items() if spec.tier == tier]`），沒有任何寫死 `"satellite_8km"` 的地方，所以拿掉這個 source 不需要動網路/module 程式碼。

## STA_H8 時間解析度：15min → 10min

`satellite_8km` 原本的 `offsets_min` 是照抄原論文 EUMETSAT 衛星規格（15 分鐘間隔、5 步、`-120..-60min`，見 `docs/rainpro_paper.md` Table 4），但 STA_H8 是台灣自己的 10 分鐘產品。已改為 `range(-120, -50, 10)`（10 分鐘、7 步，涵蓋同一個 `-120..-60min` 窗口），對應的 `TIME_TOLERANCE["satellite_8km"]` 也從 `"8min"` 改成 `"5min"`（比照其他 10-min cadence 來源）。

原論文的 `-60min` 起點是因為 EUMETSAT 有約 1 小時的 operational delay（`docs/rainpro_paper.md`: "Satellite data...has a 1-hour operational delay"）；STA_H8 是否有類似延遲尚未用真實資料核對過，`-120..-60min` 窗口本身暫時原樣保留。

## GFS：122-channel 清單（`GFS_ANALYSIS_VARIABLES`）

`gfs_variables` 原本預設是空 tuple，`include_gfs=True` 但沒手動填清單時會靜默產生 0-channel 的 `gfs_16km`。已把論文 App. I（Table 11）的 122 個 GFS channel 攤平成 canonical 名稱，設成 `gfs_variables` 的新預設值：

- 單層變數（如 `PRATE`、`GUST`）維持原始 GRIB2 代碼，不加後綴。
- 多層變數用 `VAR_層別` 命名（如 `TMP_850mb`、`TMP_surface`、`HGT_trop`），每個變數各自的層數直接照 Table 11（`TMP` 8 層、`PRATE` 1 層、`CAPE` 4 個非標準層等）。
- 這是 canonical 名稱，不保證等於實際 GFS zarr store 的欄位名，需要透過 `variable_aliases`（`RainPro8Dataset`）對應。目前完全沒有對照過真實 store schema，開始接真資料時大機率需要調整。

驗證：`len(GFS_ANALYSIS_VARIABLES) == 122`、無重複，且 `build_taiwan_sources(include_gfs=True)` 的 `gfs_16km` channel 數精確等於 122。

## Metrics 實作細節

### `probs` 的語意（CRPS/Brier 的基礎）

`rainpro.network.rainpro8.RainPro.predict()`：`preds = cumprod(sigmoid(outputs), dim=2)`，是「P(value > bucket_c.min)」，在 c 上非遞增（cumprod 本身就是 ordinal-consistency 機制）。`EvalOutputs.probs = 1 - preds` 因此是「F(bucket_c) = P(value <= bucket_c.min)」，一個在 bucket 邊界上離散化的 CDF，在 c 上非遞減。`rainpro/metrics/probabilistic.py` 直接拿 `target <= edge` 算 indicator，不透過 `Bucketize` 反推索引——比較簡單，也避免 `Bucketize` 索引慣例的 off-by-one 風險。

Tail 假設（codebase 其他地方都沒寫，這裡明確講）：低於第一個 bucket 邊界視為 F=0，高於最後一個視為 F=1。

### FSS 視窗定義與 NaN 遮罩

- FSS 的「1/2/4/8 格」鄰域，實作為正方形視窗邊長（N×N），不是半徑。
- `target` 在 QPESUMS 覆蓋範圍外是 NaN（`rainpro/data/rainpro8_dataset.py`，`keep_nan=True`）。現有 `CriticalSuccessIndex`（`rainpro/metrics/csi.py`）沒有遮罩 NaN——`NaN >= threshold` 算 False，NaN 像素會被當成「觀測無雨」計入，可能讓覆蓋缺口區域的 FAR 被抬高。**刻意決定**：不動 `CriticalSuccessIndex`（避免動到已經跑過、可能已經在比較的 CSI 數字），只有新 metrics（`ContingencyMetrics`、`FractionsSkillScore`、`LeadTimeMAEMSE`）遮罩 `~isnan(target)`。這是 CSI 與新 metrics 之間刻意保留的不一致，不是要修的 bug。

### FSS(window=1) ≠ CSI（曾經寫錯過一次）

規劃 `tests/test_metrics.py` 時原本假設「FSS window=1 應該等於同一 threshold 下的 CSI，可以互相驗證」——這是錯的。FSS = 1 - MSE/MSE_ref，在 window=1（無鄰域平滑）時逐像素化簡成 `2*hits / (2*hits + misses + false_alarms)`，跟 CSI 的 `hits / (hits + misses + false_alarms)` 是不同公式，只有在完全沒有誤判（misses=false_alarms=0）時才會重合。測試已經改成用正確的 closed-form 驗證，不跟 CSI 比較。

### 沒有 `evaluate.py` / parquet

原本規劃過一個獨立的 `evaluate.py`（吃 checkpoint list、輸出跨 arm/seed 比較用的 parquet）。使用者決定不需要——每個 run 現有的 WandB log 已經夠比較用了。所以 `rainpro/callbacks/log_plots.py::LogPlots` 改成泛化處理任何暴露 `full()`（回傳 `dict[str, Tensor]`，1D `[T]` 或 2D `[K, T]`）的 metric，用跟 CSI 一樣的三種圖（per-lead-time-per-threshold、threshold 平均、lead-time 平均）自動畫出來，`ReliabilityAccumulator` 另外用 `full_table()` 輸出成 `wandb.Table`（bucket × lead_time × bin 太多組合，不適合自動畫線圖）。跨 arm/seed 的比較留給使用者在 WandB UI 上做（用 run name/filter），沒有另外做 tagging 自動化。

## GT 重新定義：QPESUMS max dBZ nowcasting（移除 Marshall-Palmer）

**結論**：任務重新定義為 QPESUMS max dBZ 的 nowcasting，GT 直接使用原始 dBZ，訓練路徑不再做 Marshall-Palmer（MP）轉換。dBZ → mm/h 降級為 post-hoc 的輸出層重新標記（`rainpro.data.marshall_palmer` 保留，只給報告/CRPS 這類 post-hoc 用途用）。

### 為什麼

1. **只有 QPESUMS 和 CWB_GAUGE 是觀測。** RainBell 是預報產品，拿它當 GT 等於訓練模型去模仿 RainBell，技巧上限被鎖死，還會把它隨 lead time 增長的誤差一起學進去。R01 從欄位命名（`MDBZ`）、`RAIN`/`MDBZ` 的組合、以及體積（2.5 年 3.0 TB，QPESUMS 同期僅 10.7 GB）判斷幾乎確定是模式輸出——確認的話要查 store attrs 有無 WRF 欄位、`RAIN` 是否為累積量、`val` 維度的真實含義。兩者正確角色都是 baseline 或 input，不是 GT。
2. **改用 dBZ 不需要改 head。** `Bucket.size`（`rainpro/loss/ordinal_consistent.py`）從未在別處被讀取，只有 `b.min` 進 `Bucketize.bounds` 和 `Threshold.bucket_vals`；loss 是 target 分箱後的 ordinal BCE，bucket 數值只定義分類邊界，不進任何算術。`dbz_to_mmh` 嚴格單調，所以在 mm/h 空間用邊界 B 分箱，等同在 dBZ 空間用 `mmh_to_dbz(B)` 分箱，逐像素完全相同——這是換單位標籤，不是改架構。
3. **MP 的誤差不會讓模型學錯，但會讓宣稱不誠實。** Z=200R^1.6 是層狀降水假設，台灣對流/颱風降水的 DSD 差距很大（同一個 39 dBZ，MP 說 10 mm/h，熱帶型 Z=32.5R^1.65 說 28 mm/h）。加上 max dBZ 是柱狀最大值而非近地面回波、亮帶、冰雹、中央山脈東側低層遮蔽，「mm/h」這個標籤支撐不住。改用 dBZ 後輸入是回波、target 是回波、評估是回波，整條鏈自洽。
4. **原本的 mm/h bucket 邊界換算到 dBZ 空間不合理。** 18 個 mm/h 邊界換算後間距呈鋸齒狀（4.8 → 1.6 → 4.8 → 0.7 → 2.8 dB），因為原清單是分段各自等距、在對數空間變形；最密處 0.7–1.1 dB 已接近雷達量化精度，模型在那裡是在學雜訊。天花板 45.4 dBZ 也太低，台灣強對流常態到 55–60 dBZ，全被壓進同一個頂層 bin。

### 修改項目（已完成）

| # | 檔案 | 修改 |
|---|---|---|
| 1 | `rainpro/data/rainpro8_dataset.py` | 移除 target 的 `dbz_to_mmh` 呼叫；target 保持原始 dBZ |
| 2 | `rainpro/loss/ordinal_consistent.py` | `taiwan_buckets` → `taiwan_dbz_buckets`，邊界改為 `[5, 10, 15, 20, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 60]`（5 dB 間隔到 25 dBZ，3 dB 間隔到 55 dBZ，60 dBZ 封頂；未用 training set 分位數決定，先用手訂邊界） |
| 3 | `rainpro/modules/rainpro8.py` | `CSI_THRESHOLDS_MMH` → `CSI_THRESHOLDS_DBZ = [20, 25, 30, 35, 40, 45]` |
| 4 | `rainpro/data/marshall_palmer.py` | 不變，但移出資料路徑，改供 post-hoc 標記使用（例：`rainpro/metrics/probabilistic.py` 的 CRPS mm/h 積分權重） |

架構、`out_channels`、loss 形式、`OptimalThresholds` 流程皆不變（`taiwan_dbz_buckets` 一樣是 16 個 bucket 的清單，只是換了名字跟數值）。

### 影響範圍

| 指標 | 換到 dBZ 後 |
|---|---|
| CSI / FSS / FBI / POD / FAR | **不變**（純門檻二值化，單調變換下不變） |
| CRPS | **積分權重換算成 mm/h 報告**：分類/indicator 仍在 dBZ 空間比對（`target <= dBZ 邊界`），但積分權重（bucket gap）用 `dbz_to_mmh` 把 dBZ 邊界換算成 mm/h 後取差值——dBZ 是對數尺度，直接在 dBZ 空間積分會不成比例地壓縮高強度端的誤差權重。見 `rainpro/metrics/probabilistic.py` 的 `_bucket_gaps_mmh` |
| Brier / reliability | 維持 dBZ（不做積分，沒有這個問題），bucket 標籤直接是 dBZ 值 |
| MAE / MSE | **會變**，直接在 dBZ 數值軸上算，不換算——尚未有需求要求換算回 mm/h |

**對 B/C 主實驗完全無影響**——三臂共用同一 GT 定義，相對比較的效力不受任何影響。

## QPESUMS -99 的語意錯置 + loss 的 NaN 遮罩失效（兩個連動的 bug，已修）

### 症狀

推論視覺化時，模型在**沒有回波的整片背景**輸出 60 dBZ（色階頂端），而且深紅區域精準對應 GT 的
NaN 區域。

### 根因一：loss 的 `nan_mask` 比對錯對象

`OrdinalConsistentLoss.forward` 原本：

```python
nan_mask = targets == self.no_data_value   # ← 在 bucketize 之前，用原始 dBZ 比對
targets = self.bucketize(targets)          # ← NaN 在這一行才變成 class 16
```

`no_data_value` 是 `Bucketize` 指派給 NaN 的**類別索引**（`= len(buckets) = 16`），不是 dBZ
值。拿原始 dBZ 去比對它（`nan == 16`）永遠是 False，所以缺測像素從未被標記。接著
`bucketize` 把它變成 class 16 → `targets_encoded` 全為 1 → `sets_mask` 全為 True →
**16/16 個 channel 都被監督成「超過所有門檻」，即模型被明確訓練成「沒有資料的地方就輸出最大回波」**。

上游 SEVIR 版本同樣有這個順序問題，但 SEVIR 的 raster 沒有 NaN，所以一直是休眠的；台灣版因為
QPESUMS 用哨兵值表示缺測而引爆。

修法：`nan_mask = torch.isnan(targets)`。

### 根因二：-99 被當成缺測，但它其實是「無回波」

見 `docs/rainpro_dataset.md` 的 QPESUMS 段落。`QPESUMS_MISSING_VALUES` 原本包含 -99，使得
**98.4% 的像素變成 NaN**，等於丟掉幾乎全部「這裡沒有下雨」的負樣本。

判定依據（三項獨立證據）：

1. -99 的遮罩隨天氣變化 —— 相隔數年的兩個時間點只有 **87.6%** 重疊
2. 推論圖上 GT 的非 NaN 區域形狀會跟著回波移動，不是固定的地理遮罩
3. 論文 Table 8：訓練集 **79.64%** 的像素落在最低的「無雨」bucket 且**是有監督的類別**，
   missing 僅 **12.97%**。把 -99 當缺測會讓台灣版 missing 變成 98.4%，與論文設計不符

修法：`SourceSpec` 新增 `no_echo_values` / `no_echo_fill`，把哨兵值依語意拆開：

| 哨兵值 | 語意 | 處理 |
|---|---|---|
| -999 | 真正未觀測 | → NaN → 排除於 loss |
| -99 | 無回波（有觀測） | → 0 dBZ → class -1 → 監督 `P(Y > 5 dBZ) = 0` |

### 為什麼必須兩個一起修

| target 像素 | 修之前 | 只修根因一 | 兩個都修 |
|---|---|---|---|
| -99（98.4%） | 訓練成 60 dBZ | 被排除，無梯度 | 訓練成「無回波」✅ |
| -999 | 訓練成 60 dBZ | 排除 ✅ | 排除 ✅ |
| 30 dBZ | 7/16 channels | 7/16 ✅ | 7/16 ✅ |

只修根因一的話，模型只會在 1.6%（全是有回波）的像素上訓練，從「被教成最大回波」變成
「沒學過哪裡不下雨」，一樣會到處長回波。

### 影響範圍

- **所有在此修正前訓練出的 checkpoint 全部作廢**，必須重練
- **舊的 val/test 指標不能與新的比較**：先前 CSI/FSS/CRPS/Brier 的分母只涵蓋非 NaN 的像素
  （約 1.6% 的畫面），修正後涵蓋整個 canvas，數字的意義完全不同
- 輸入端（`radar_4km` / `radar_8km`）幾乎不受影響：-99 原本走 NaN → `fill_value=0.0`，
  修正後走 0 dBZ → `minmax_normalize` 後為 0.0154（`DBZ_RANGE = (-1, 64)`），差異可忽略

## 訓練吞吐相關的實作決策

本節的每個決定都有對應的量測，重現方式見 `scripts/profile_training_pipeline.py`（各段以
`--sections` 指定）。參考點是論文 Table 9 的 **0.490 秒 / optimizer step**（100k steps、
batch 16、單張 H100 SXM5 80GB + 26 vCPU，`docs/rainpro_paper.md:267` 與 `:626`）。本專案在單張
H200 上以 batch 4 × accumulate 4 量到 **T_real = 0.747 秒 / step**，其中 GPU-only 為 0.679 秒。

### 網路寬度：`dims = (256, 256, 256, 256)`

論文 Sec. 4 明寫「We use 256 channels throughout the entire network, totaling 36.7 million
parameters」，Sec. 3.4 則描述相對 MetNet-3 的 227M 做了「halving internal channels」。
`rainpro/network/rainpro8.py` 的 `dims` 是 `(dim_4km, dim_8km, dim_16km, dim_2km)`，其中
`dim_16km` 同時是 MaxViT centre 的寬度（`MaxVitBlocks(in_channels=dim_16km, ...)`）。

實測各組態的參數量（`--sections model`）：`(128, 256, 512, 128)` 為 **82.62M**，其中 MaxViT
一個模組就佔 64.811M（**78.4%**）；平坦的 `(256, 256, 256, 256)` 為 **≈36.7M**，與論文一致。
上游沒有附 RainPro-8 的訓練設定檔可以仲裁（repo 內唯一的 `config.yml` 是 SEVIR 的，而其姊妹網路
`rainpro/network/rainpro.py` 採用單一純量 `dim: int = 256`），因此以論文敘述為準。

MaxViT 的 FLOPs 與參數量大致隨 `dim_16km²` 成長，所以這同時是精度與算力的決定，不只是尺寸對齊。

### `torch.compile`：rebind `RainPro.forward`

`RainPro8Module.__init__` 的 `compile_model`（預設 `True`）做的是：

```python
self.model.forward = torch.compile(self.model.forward)
```

**刻意不用 `torch.compile(self.model)`**，兩個理由都是必要條件：

1. `RainPro.predict()` 內部呼叫的是 `self.forward(...)`。`torch.compile(module)` 只會編譯
   `__call__`，`predict` 這條實際被走的路徑仍然是 eager。rebind 綁定方法則讓 `self.forward`
   直接解析到編譯後的函式。
2. `torch.compile(module)` 回傳 `OptimizedModule`，會把所有參數名加上 `_orig_mod.` 前綴，
   checkpoint 因而與未編譯的 run 不相容。rebind 不更動 `state_dict()` 的 key，所以
   `compile_model` 開或關的 checkpoint 可以互換，`scripts/infer_visualize.py` 也不需配合修改。

`self.model.criterion`（`Bucketize` / `Threshold`）留在圖外。

量測（H200、`dims=(256,)*4`、batch 4 × accum 4、fp32）：關閉為 2.325 秒/step、峰值 41.7 GiB，
開啟為 **0.679 秒/step、27.0 GiB**（3.4×）。

收益幾乎全部來自一個病灶：`rainpro/network/clt.py` 的 `LayerNorm`（非 attention 分支）使用
`nn.GroupNorm(num_groups=1)`，而 PyTorch 的 CUDA 實作把 moments kernel 的 grid 開成
`N × num_groups` 個 block —— micro-batch 4 時只有 **4 個 block 跑在 132 個 SM 上**。該 kernel
單次 5.4 ms、佔全部 CUDA 時間的 **59%**（`--sections kernels`）。Inductor 會把
`aten.native_group_norm` 分解成 `var_mean` 加 elementwise 再融合，該 kernel 隨之從 profile 中
消失，榜首回到正常的 `convolution_backward` 與 `aten::mm`。

因為瓶頸是佔用率而非算力，精度旗標在此之前幾乎無效（TF32 僅 1.12×，bf16 與 TF32 打平）；
`rainpro/network/clt.py` 另外保留一個等價的 `GroupNorm1VarMean`（以 `set_norm_impl("var_mean")`
切換，`--sections checknorm` 驗證 forward 與三條 gradient path 的一致性），作為 `torch.compile`
不可用時的備案，預設不啟用。

### zarr 開啟方式：`chunks=None`

`rainpro/data/rainpro8_dataset.py::_open_store` 統一以 `xr.open_zarr(..., chunks=None)` 開啟所有
store，亦即走 xarray 自己的 lazy indexing 而非 dask。

訓練的讀取樣態是「每個 sample 62 個 frame、每個 frame 一個 chunk、明確索引、同步取用」，這正是
dask 最不擅長的形狀。實測讀取 36 個 QPESUMS frame（`--sections readpath`）：

| 方式 | 秒 / 次 |
|---|---|
| xarray + dask | 0.5685 |
| xarray、`chunks=None` | **0.0224**（25.4×） |
| 直接索引 zarr array | 0.0229 |

`chunks=None` 與「完全繞過 xarray 直接讀 zarr」打平，這一點是判定差異來自 dask 而非儲存層的關鍵
證據。另外兩項證據：以相同索引重跑一次讀取（page cache 命中）時間不變，排除檔案系統延遲；成本隨
**chunk 數**而非位元組量變化（衛星讀取 2.5 倍的位元組卻只花 0.6 倍時間），排除解壓縮。換算下來
dask 在每個 chunk 上收取約 24 ms 的 scheduler 與 graph 建構開銷，而實際讀取加解壓約 0.6 ms。

dask 另有一個在 SLURM 上特別傷的性質：它的 thread pool 大小取自 `os.cpu_count()`，回報的是整台
節點而非 cgroup 配額，因此 `num_workers=11` 會產生約 1000 條執行緒搶 12 個核心。這是真實訓練中
p95 達 9.4 秒的停頓來源；移除 dask 後 p95 降至 **0.843 秒**（改善幅度大於平均值的改善，與此機制
一致）。

移除 dask 後，`_load_frames` 的成本已隨 frame 大小等比成長（QPESUMS 409 MiB/s、STA_H8
667 MiB/s），代表剩下的是真正的 zstd 解壓，沒有可再榨的額外開銷。

### STA_H8 多 store：時間路由，不做 concat

`scripts/compress_sta_h8_taiwan.py --freq quarter` 會把一年切成數個 store（`/home` 與 `/work`
各有 100 GB 配額，單一 store 放不下）。`data_root["sta_h8"]` 因此接受逗號分隔的多個路徑。

`_StoreHandle` 不把這些 store 串接起來，而是只合併**時間索引**，並記錄每個全域位置由哪個
dataset 擁有（`_owner`）、在該 dataset 內的索引是多少（`_local`）。讀取時依 owner 分組，每個
dataset 發一次 `isel`，因而保留 `_load_frames` 一次解析所有 offset 的批次讀取性質。36 個 offset
只橫跨 6 小時，通常落在同一季度內；跨越接縫時單純變成兩次讀取。

**不能用 `xr.concat`**：它只有在陣列是 dask-backed 時才是 lazy 的。搭配 `chunks=None` 時它會呼叫
`np.concatenate`，在第一次 `_get_store` 就把全部約 121 GB 拉進記憶體。

合併索引以 stable sort 排序，確保 `get_indexer(..., method="nearest")` 所要求的單調性成立，
與路徑列出的順序無關。

路由錯誤會回傳來自錯誤季度、但外觀完全合理的資料，在 loss 曲線上看不出來，因此
`--sections boundary` 專門檢查這件事：找出所有 ownership 變換點，在每個接縫上讀一個跨界窗口，
與「逐一從擁有它的 store 單獨讀」的結果逐 frame 比對，另加一個內部窗口確保該路徑確實被執行。

### `eval_batch_size` 繼承 `batch_size`

`rainpro8.yml` 刻意不設定 `data.eval_batch_size`，讓 `RainPro8DataModule` 回退到 `batch_size`。

網路經過 `torch.compile` 之後，不同的 eval batch 等同不同的輸入形狀，Inductor 會在每次進入
validation 時重新編譯整張圖；`check_val_every_n_epoch: 1` 代表每個 epoch 各付一次數分鐘的代價。
留空可確保即使只在 CLI 覆寫 `--data.batch_size`，兩者仍然一致。

### precision 維持 `'32'`

與論文一致，而且在目前的配置下切換到 bf16 沒有 end-to-end 收益：bf16 會把 GPU-only 降到
0.446 秒/step，相當於需要 35.9 samples/s 的資料供應，而 11 個 worker 實測供應
19.07 samples/s（`--sections workers`）。資料端封頂時，GPU 端更快並不會反映到 wall clock。
等資料端再快一輪之後才值得重新評估。

### 已量測但刻意延後的項目

| 項目 | 量測 | 延後的理由 |
|---|---|---|
| QPESUMS 改用解析式 regrid | `regrid_prepare_kdtree` 佔每 sample 0.117 秒（25.8%），其中 4 次呼叫有 3 次是 QPESUMS | QPESUMS 是 0.0125° 規則網格，最近鄰可純算術求得，不需 `cKDTree`；預期 T_real 0.747 → 約 0.68 秒，115k steps 僅省約 2 小時 |
| `channels_last` | 編譯後的 profile 中 `nchwToNhwc` 與 `nhwcToNchw` 合計 5.96% | 需確認所用 PyTorch 版本的 GroupNorm 是否有 NHWC CUDA 路徑，否則只是多一次 layout 轉換 |
| micro-batch 16 | `dims=(256,)*4` 未編譯時於 140 GiB 上 OOM | 編譯後峰值降至 27.0 GiB，已有空間重測，但資料端封頂時增益有限 |

訓練期間再處理，不阻擋第一輪正式訓練。
