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
