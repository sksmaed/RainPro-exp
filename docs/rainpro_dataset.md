# RainPro-tw 對應資料描述

## 資料對應清單盤點

| 資料 | RainPro-8 對應資料 | 空間範圍&解析度 | 時間解析度 |
| ----- | ----- | ----- | ----- |
| QPESUMS max dBZ | target_2km (0 - 6h) | 384 × 512 km / 2 km | 10 min |
| QPESUMS downsampling | radar_4km (−60…0 min) | 565 × 780 km / 4 km | 10 min |
| QPESUMS downsampling | radar_8km (0 min) | 565 × 780 km / 8 km | - |
| STA_H8 | satellite_8km (−120…−60 min） | 1536 × 1536 km / 8 km | 15 min |

## 資料集描述

### QPESUMS

- 型態：雷達
- 來源性質：回波觀測
- 時間解析度：10 分鐘
- 空間解析度 / 網格：561 × 441 @ 0.0125°（台灣）
- 欄位：max dBZ、time、經緯度
- 缺測值：-999 or -99

### STA_H8

- 型態：衛星（靜止軌道）
- 來源性質：觀測
- 時間解析度：一小時
- 空間解析度 / 網格：2750 × 2750 LCC grid（STA_H8_Plt_IR_for_glbdisplay 有逐像素經緯度查找表）
- 欄位：9 個紅外線波段（B08 - B16）亮溫

## 預計壓縮策略

| 資料集 | 壓縮器 | 整數化 | Chunk | Schema | 其他處理 | 壓縮率 | 2.5 年原始 | 2.5 年壓縮後 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **CWB_AUTOSTA** | `zstd-5 BITSHUFFLE` + Delta | **int16**，逐欄 scale（1/10/100） | `(502, 26208)`≈ 25 MiB | `(station, time)`；規則 10 分鐘網格 | 9 靜態欄抽測站表；9 家族保留碼；排除 SUN/TIME；OBS_TIME → STALE_MIN | **30.1x** | 11.5 GB | **0.38 GB** |
| **CWB_GAUGE** | `zstd-5 SHUFFLE` | **int16/int32**，scale=10 | `(1011, 13104)`≈ 25 MiB | `(station, time)`；規則 10 分鐘網格 | 10 靜態欄抽測站表；−998 = 六小時無降雨（**映射 0 非 NaN**）；TIME = 延遲步數需保留 | 60–120x（年化） | 20.5 GB | **0.25 GB** |
| **QPESUMS** | — | — | `(1, 561, 441)` 現況 | `(time, lat, lon)` | **已是 zarr v2，不處理** | — | 10.7 GB | **10.7 GB** |
| **R01** | `zstd-5 SHUFFLE` | **int16 × 0.1**（相對 ASCII 無損） | `(48, 1, 330, 330)` ≈ 20 MiB | `(time, val, lat, lon)` | 存 int + `scale_factor` attrs（不用 FixedScaleOffset，讀取慢 4.7x）；fill 逐變數（MDBZ −30 / RAIN 0） | **16.8x** | 3.0 TB | **179 GB** |
| **imerghh_full** | `zstd-5 **NOSHUFFLE**` | float32 原樣 | `(12, 450, 900)` ≈ 18.5 MiB | `(time, lat, lon)` **轉置**（原生為 lon,lat） | fill −9999.9 原樣保存；核心 3 變數 | **1.20x** | 328 GB | **274 GB** |
| **FS7_RO** | `zstd-5 SHUFFLE` | `Bend_ang`float64→float32（位元級可逆） | `(256, 3904)` ≈ 3.8 MiB | `(profile, level)`；**padded 補齊** | 月聚合（504 萬檔 → 240 store）；144 屬性 → per-profile 表 | ~2.8x | 4.38 TB | **1.56 TB** |
| **STA_H8** | `zstd-5 **NOSHUFFLE**`⚠️未測 | - | `(1, 1, 2750, 2750)` ≈ 28.8 MiB | `(time, band, y, x)` | 無缺測值（唯一）； | **~3.4x**推估 | 5.43 TiB | **~1.6 TiB** |
| **合計** |  |  |  |  |  | **~3.6x** | **~13.1 TB** | **~3.6 TB** |