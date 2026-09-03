# RainPro-8-TW 評估架構

本文件定案 RainPro-8-TW（QPESUMS + STA_H8 + 選用 GFS）的實驗臂設計與分階段評估計劃。核心問題：**在台灣、以 QPESUMS 為 ground truth 的設定下，16km NWP（GFS）tier 是否對 0–6h 降雨預報帶來可歸因於「資訊」而非單純「架構容量」的增益？**

實作層級的細節（程式碼為什麼長這樣、具體數學推導、bug 修正）記錄在 `docs/rainpro_tw_implementation_notes.md`；本文件談完整的實驗設計與分析方法，包含沿用自 RainPro-8 原論文的通用評估原則，以及台灣資料管線特有的部分（內文以「TW 版新增」標註）。

## 實驗臂

| 代號 | 內容 | Config | lead 範圍 | 角色 |
| --- | --- | --- | --- | --- |
| A | RainPro-8-TW radar-only | `include_satellite=false`, `include_gfs=false` | 0–6h | 來源消融起點 / sanity floor |
| B | RainPro-8-TW obs-only（QPESUMS + STA_H8） | `include_satellite=true`, `include_gfs=false` | 0–6h | 主對照組 |
| C | RainPro-8-TW obs+GFS | `include_satellite=true`, `include_gfs=true` | 0–6h | 主實驗組 |

三臂共用同一份 GT（`target_2km`）、同一份 test 時間清單、同一套評估 metrics（見 Stage 2）與同一支評估流程——差異只在 `data.include_satellite`/`data.include_gfs` 這兩個 flag（`rainpro8.yml`、`rainpro/data/rainpro8_sources.py::build_taiwan_sources`，**TW 版新增**）。所有臂必須走同一份 test 索引、同一套 metric 程式，否則臂與臂之間的差異可能來自評估細節而非模型本身。

A 的 `include_satellite=False` 為什麼不影響 `radar_8km`（兩者共用同一個 8km tier）的實作細節見 `docs/rainpro_tw_implementation_notes.md`。

## 分階段計劃

### Stage 0：SEVIR 冒煙測試

目標：確認 OCL（Ordinal Consistent Loss）與整體訓練管線是健全的。這一步跑的是既有的 `RainPro-2R`（`rainpro2r.yml` → `rainpro.modules.sevir.SEVIRModule`），與 Taiwan 資料管線完全無關，是共用架構本身的健全性檢查——後面所有台灣的數字都建立在這份實作是對的之上，這一步失敗，後面全是白工。

做法：`python main.py fit rainpro2r --config rainpro2r.yml`，跑幾百 step 確認 loss 下降、test 能跑完、CSI 落在論文量級（~0.35）。不寫評估腳本、不進報告表。

### Stage 1：資料與 GT 定案

目標：把「GT 是什麼」「16km tier 是什麼」寫死，之後不再動。

- **GT 必須是觀測，不能是另一個模式/預報產品**：`target_2km` = QPESUMS max dBZ，**直接使用原始 dBZ，訓練路徑不做 Marshall-Palmer 轉換**（任務定義為 dBZ nowcasting；完整理由、bucket 邊界推導見 `docs/rainpro_tw_implementation_notes.md`，**TW 版新增/調整**，原論文的 mm/h 定義見下方對照）。三臂共用、不再更動。拿另一個模式的輸出（例如預報產品）當 GT，等於訓練模型去模仿那個模式，技巧上限被鎖死，還會把它隨 lead time 增長的誤差一起學進去——這是不分台灣或原論文都成立的通用原則。
- **CWB_GAUGE 比對（選用，附錄用）**：GT 本身是 dBZ，不需要靠 gauge 驗證「mm/h 對不對」；但 post-hoc 報告轉 mm/h（例如 CRPS 的積分權重，見 implementation notes）目前用的是通用 Marshall-Palmer 係數。若要提升這個轉換在台灣的準確度，可選用 CWB_GAUGE 建立本地 Z-R 關係，但這只影響報告可讀性，不影響 dBZ 空間的主結論。目前 repo 內沒有這支腳本，需要從零寫（**TW 版新增**）。
- **16km tier 內容**：沿用原論文的作法——真實 GFS，選用論文 App. I 的 122-channel 選擇。在 TW pipeline 中，這份清單本身沒有現成程式碼，已重建為 `GFS_ANALYSIS_VARIABLES`（`rainpro/data/rainpro8_sources.py`），命名規則為 `VAR` 或 `VAR_層別`（如 `TMP_850mb`）；這些是 canonical 名稱，須透過 `variable_aliases`（`RainPro8Dataset`）對應到實際 GFS zarr store 的欄位名（**TW 版新增**：清單重建與 store 對接）。
- **GFS normalization**：`rainpro/data/normalize.py` 的 `DEFAULT_NORM_BOUNDS` 只涵蓋雷達 dBZ 與 STA_H8 紅外線亮溫，GFS 變數完全未正規化。**這一步不做，C 臂必然表現差，會得到假的「GFS 沒用」結論**——必須先算出 122 個 GFS 變數在 training set 上的 min/max/mean/std，填進 `data.norm_bounds` 才能開始 Stage 3 的 C 臂訓練（**TW 版新增**：計算腳本待寫）。

### Stage 2：評估基礎建設

目標：把「一個 checkpoint → 一份完整評估結果」變成可重複的流程，三臂共用同一套 metrics、同一份 test 索引。

三臂共用同一套 metrics（`rainpro/modules/rainpro8.py::create_metrics`），每個 `test` run 透過 `rainpro/callbacks/log_plots.py::LogPlots` 把完整的 per-lead-time（部分還有 per-threshold）breakdown 送進 WandB——跨 arm/seed 的比較在 WandB UI 上做（依 run name / filter）。單獨一個「CSI = 0.28」的數字本身沒有意義，只有「比 A 高多少、加了 GFS 之後比 B 高多少」才是結論。

| 指標 | 為什麼要它 | 狀態 |
| --- | --- | --- |
| CSI @ [20, 25, 30, 35, 40, 45] dBZ | 主指標。門檻二值化，直觀、業界慣用 | 已有（`rainpro/metrics/csi.py`） |
| FSS（鄰域 1/2/4/8 格） | 容忍位移。CSI 會把「對但偏一格」當全錯，加了 NWP 之後場位移常見，只看 CSI 會誤判 | 已建（`rainpro/metrics/fss.py`，**TW 版新增**） |
| CRPS | 機率品質，論文消融表的主指標。模型輸出是 bucket 化的累積機率，只看 CSI 等於把機率丟掉 | 已建（`rainpro/metrics/probabilistic.py`，**TW 版新增**） |
| Brier / reliability diagram | 機率校準：「說 70% 的格點是不是真的 70% 下雨」 | 已建（同上，`BrierScore`/`ReliabilityAccumulator`，**TW 版新增**） |
| FBI + POD/FAR 分開 | 診斷用。CSI 掉了要能回答是漏報還是空報 | 已建（`rainpro/metrics/contingency.py`，**TW 版新增**） |
| MAE/MSE | 參考用，但對 no-rain 主導的分布不敏感，只當附錄 | 已建（`rainpro/metrics/regression.py`，**TW 版新增**） |

一個實作注意：`CriticalSuccessIndex` 是對全 test set 累積 hits/false_guesses 再算比值（pooled contingency table，這是對的，不要用「每個 batch 算一次 CSI 再平均」的寫法，那樣會被稀有事件的 batch 嚴重扭曲）。但 `compute()` 預設會對 lead time 和 threshold 都取平均——不要用這個平均值當主數字，台灣冬季高強度事件極稀少，某些 lead time 的 CSI 可能是 0，把平均拉爛。主表報 per-threshold、per-lead-time，附錄再報平均。新增的 metrics 都遵循同一個 pooled 累積模式。

閾值：`rainpro/network/optimal_threhsolds.py`（60 個候選值只在 val grid search、對 test 套用、per-checkpoint-dir 快取 `best_thresholds.pt`，threshold 只在 val 上優化，不要動）行為不變，`--thresholds` 直接沿用。

尚未執行（不是程式碼問題，是還沒接上真實資料/環境跑過）：`tests/test_metrics.py` 的 hand-computed 正確性測試還沒在有 `torch` 的環境跑過；GFS `norm_bounds` 計算腳本、CWB_GAUGE Z-R 比對腳本都還沒寫。

### Stage 3：訓練矩陣

目標：產出可比較的 checkpoint。

- 每臂（A/B/C）3 個 seed（`seed_everything`）。`seed_everything` 已經在 `cli.py`/`cli_rainpro8.py` 用 `seed_everything_default` 接好，跑的時候用 `--seed_everything <N>` 覆蓋即可，不用改程式碼。
- 相同 max_epochs / split（`start_date: 2023-12-01` ~ `end_date: 2024-11-30`，12/2/2 cycle，12h blackout，`rainpro8.yml` 現有設定，**TW 版新增**：`cycle_split` 這種帶 blackout buffer 的多天週期切分）。
- 記錄參數量、峰值 VRAM、單 batch 推論時間（三臂並列，供 Stage 4.5 效率分析）。
- test 一律加 `--thresholds`，threshold 只在 val 上優化。
- 3 seed × 3 臂 = 9 次訓練。若算力不夠，優先順序是 B、C 各 3 seed（這是核心問題），A 用來當 sanity floor，可以先跑 1 個 seed 定位。
- **為什麼要 3 seed**：單一 seed 的 CSI 差 0.005 完全可能只是初始化雜訊，沒有多 seed，無法宣稱 GFS 有沒有用——這是通用的統計原則，不分台灣或原論文。

### Stage 4：評估分析

**4.1 主圖：CSI/CRPS vs lead time（36 個 lead time 分開畫，不平均）**

這是整個實驗最關鍵的一張圖。原論文的 attribution 分析（Integrated Gradients）明講：近期高解析雷達主導短 lead，低解析雷達在 4 小時左右開始有價值，GFS 變數在長 lead 才變重要，GFS forecast 對前 4 小時幾乎沒有貢獻。所以「平均 CSI」很可能把 GFS 的效果稀釋到看不見，36 個 lead time 一定要分開畫。

| 觀察 | 預期 | 若不符合代表什麼 |
| --- | --- | --- |
| 0–1h，B vs C | 幾乎重疊 | C 明顯優於 B → 高度懷疑洩漏，檢查 `gfs_forecast_16km` 的 offsets 與 init time 對齊，確認沒有把 valid time ≤ 0 的資料餵進去 |
| 3–6h，B vs C | C > B，ΔCSI 約 0.01–0.03（**待 C-placebo 排除容量混淆前，先當「obs+GFS 設定整體優於 obs-only」，不直接歸因 GFS 資訊**，見 Future Work） | 沒有分岔 → 先排除正規化/時間對齊/變數選錯，再下結論 |
| 全 lead，A vs B | B > A，且差距隨 lead 拉大 | 衛星的價值應在 3h 後才顯現（雲頂資訊 vs 雷達外延失效） |
| P0/P1 級別的傳統方法 vs 所有 DL 臂 | 1h 後 DL 全面勝出 | 若目前沒有這類 baseline 可比較，至少確認 A 臂本身遠優於「t=0 凍結」這個直觀下限 |

**4.2 機率品質**：CRPS vs lead time、reliability diagram（分 threshold）。加 GFS 有可能 CSI 幾乎不動但 CRPS 和校準明顯改善——這是有意義的正面結果，只看 CSI 會漏掉。原論文消融表裡 RainPro-8 vs radar-only 的 CRPS 差距（0.06096 vs 0.06574）比 CSI 差距更早顯現，值得優先看。

**4.3 偏差診斷**：FBI、POD、FAR 各自 vs lead time。如果 C 的 CSI 上升，要能說出是「抓到更多真事件」還是「亂噴更多」。加 GFS 常見效果是場變模糊 → POD↑、FAR↑、FBI 往 >1 偏。論文報 RainPro-8 的 FBI 是 1.262（略微過度預報）；若 C 臂 FBI 衝到 1.6 以上，CSI 的增益要打折看待。

**4.4 分層（選用，需要外部標籤，**TW 版新增**）**：把 test cases 依下列切分後重算全部指標——天氣型態（颱風/梅雨/午後對流/冬季東北季風，台灣特有分類）、強度（per-threshold 已經分了，再加上事件日 vs 無雨日）、地形（山區/平原/海面）、日夜（STA_H8 的 IR 波段日夜都可用，但雲頂資訊在不同時段價值不同）。**repo 內沒有這些標籤資料**（案例日期清單、地形遮罩），需要使用者提供權威的個案日期清單，不要把「颱風/梅雨季」的假設寫死在程式碼裡。全年平均會把訊號洗掉：台灣一年裡真正有大雨的時數佔比很低，GFS 的價值最可能集中在有明確綜觀強迫的梅雨/颱風個案，午後對流（局地、快速生消）幾乎沒用——這個 pattern 本身就是一個好結果，比平均值有價值得多。

**4.5 效率**：參數量、VRAM、推論時間、訓練時間，三臂並列。C 多了一整個 16km encoder tier，若 ΔCSI 只有 0.005 但推論成本增加 30%，結論應該是「不值得」。

### Stage 5：統計與結論

對 test cases 做 paired bootstrap（重抽個案而非格點——格點之間空間相關，格點層級的 bootstrap 會嚴重低估變異），給 ΔCSI/ΔCRPS 的 95% CI。主表格式：每格填 mean ± std（3 seeds），關鍵比較另外標 CI 是否跨 0。

C-placebo 補上後（見 Future Work），Stage 5 的目標結論句才能完整成立：「在台灣、0–6h、以 QPESUMS 為 GT 的設定下，加入 16km NWP tier 在 3 小時之後帶來 ΔCSI = X ± Y，且該增益不能由架構容量解釋（vs placebo）」。**在此之前，只能宣稱「obs+GFS 優於/不優於 obs-only」，不能宣稱「GFS 資訊有用」**。

## Flag / 設定速查（皆為 TW 版新增）

| Flag | 位置 | 預設 | 用途 |
| --- | --- | --- | --- |
| `data.include_satellite` | `rainpro8.yml`, `RainPro8DataModule` | `true` | `false` → A（radar-only），`satellite_8km` 移出 `sources`，`radar_8km` 不受影響 |
| `data.include_gfs` | `rainpro8.yml`, `RainPro8DataModule` | `false` | `true` → C（obs+GFS），16km tier 出現 |
| `data.gfs_variables` | `rainpro8.yml`, `RainPro8DataModule` | `GFS_ANALYSIS_VARIABLES`（122 channel） | 覆寫 GFS 分析場欄位清單 |
| `data.variable_aliases` | `rainpro8.yml`, `RainPro8DataModule` | `None` | 把 canonical 變數名對應到實際 zarr store 欄位名（STA_H8、GFS 都可能需要） |
| `data.norm_bounds` | `rainpro8.yml`, `RainPro8DataModule` | 只有 dBZ/IR 有預設 | Stage 1 算完 GFS 統計量後填入，否則 GFS 變數不正規化 |
| `--thresholds` | CLI (`main_rainpro8.py test`) | 關 | 開啟後套用 `optimal_threhsolds.py` 在 val 上優化的門檻 |
| `--seed_everything <N>` | CLI | `0` | Stage 3 訓練矩陣的 3-seed 依此覆寫 |

## Future Work

- **C-placebo**：把 `gfs_16km`/`gfs_forecast_16km` 的通道換成常數或時間上隨機打亂的值（架構完全不變，只抽掉資訊），確認 B vs C 的增益究竟是「GFS 資訊」還是單純「多了一個 encoder 分支的容量」。做出來之前，B vs C 只能宣稱「obs+GFS 優於/不優於 obs-only」，不能宣稱「GFS 資訊有用」。

## 已知風險 / 注意事項彙整

- **洩漏檢查**：C 臂在 0–1h 若明顯優於 B，優先懷疑 `gfs_forecast_16km` offset 對齊問題，不要直接當成正面結果。
- **GFS 欄位命名未經驗證**：`GFS_ANALYSIS_VARIABLES` 是依論文 Table 11 建的 canonical 命名，尚未對照過任何實際 GFS zarr store 的 schema，Stage 1 開工時大機率需要靠 `variable_aliases` 調整，或發現部分變數在實際 store 裡缺失。
- **STA_H8 時間解析度窗口**：`-120..-60min` 的窗口沿用原論文 EUMETSAT 衛星的規格，是否符合實際 STA_H8 產品的可用延遲需要在 Stage 1 用真實資料核對（詳見 implementation notes）。
- **C-placebo 缺席**：見上方 Future Work。
