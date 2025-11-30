import cv2, numpy as np, matplotlib.pyplot as plt
from scipy.signal import savgol_filter, get_window
from numpy.fft import rfft, rfftfreq

# ==== ユーザ設定 =========================================================
pixel_pitch_mm = 0.00345      # センサ画素ピッチ [mm]（例：3.45µm）
theta_deg =3.3#85.8        #-2      # 既知のエッジ角度（x軸基準、反時計回り +）
oversample = 4                # ESFのサブピクセル分割
sg_poly = 3                   # Savitzky–Golay 多項式次数
sg_win  = 51                  # Savitzky–Golay 窓長（奇数）
use_click = True            # True: 画像からクリックで点を取る / False: 手入力
img_path = R"file_pass"        # 入力画像（グレースケール推奨）
# ========================================================================

# --- 画像読込 ---
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(img_path)
h, w = img.shape

# --- ROI を対角 2 点で指定 ---
if use_click:
    plt.figure(); plt.imshow(img, cmap='gray'); plt.title("click 2 point to select ROI"); plt.axis('off')
    roi_pts = plt.ginput(2, timeout=0)  # [(xA,yA),(xB,yB)]
    plt.close()
    (xA,yA),(xB,yB) = roi_pts
else:
    # 手入力例（必要に応じて書き換え）
    xA,yA = 924,668
    xB,yB = 1124,868

x0, x1 = int(min(xA,xB)), int(max(xA,xB))
y0, y1 = int(min(yA,yB)), int(max(yA,yB))
roi = img[y0:y1, x0:x1].astype(np.float32)
hR, wR = roi.shape
if hR < 16 or wR < 16:
    raise ValueError("ROI is too smal")

# --- エッジの一点指定（ROI内座標） ---
if use_click:# or not use_click:
    plt.figure(); plt.imshow(roi, cmap='gray'); plt.title("click edge point"); plt.axis('off')
    pt = plt.ginput(1, timeout=0)
    plt.close()
    (xe, ye) = pt[0]
else:
    # 手入力例（ROI内の座標）
    xe, ye = wR/2, hR/2

# --- 角度θ（既知）+ 1点 から直線の法線を定義 ---
# 直線の接線方向 t=(cosθ, sinθ) → 法線 n = (+sinθ, -cosθ)
theta = np.deg2rad(theta_deg)
tx, ty = np.cos(theta), np.sin(theta)
nx, ny =  ty, -tx                       # 法線ベクトル（単位）
# 直線方程式: nx*x + ny*y + d0 = 0 を、点 (xe,ye) が乗るように
d0 = -(nx*xe + ny*ye)

# --- 法線距離マップ d(x,y) を作成（ROI座標系） ---
yy, xx = np.mgrid[0:hR, 0:wR]
d = nx*xx + ny*yy + d0   # 符号付き距離 [pixel]

# --- ESF（距離ビン平均） ---
d_min, d_max = np.percentile(d, [1, 99])
bins = max(int((d_max - d_min)*oversample), 128)
idx = np.clip(((d - d_min)/(d_max - d_min)*(bins-1)).astype(np.int64), 0, bins-1)
sum_vals = np.bincount(idx.ravel(), weights=roi.ravel(), minlength=bins)
count    = np.bincount(idx.ravel(), minlength=bins)
ESF = sum_vals / (count + 1e-9)


# 正規化 & 軸
ESF = (ESF - ESF.min()) / max(np.ptp(ESF), 1e-9)
grid = np.linspace(d_min, d_max, bins)             # [pixel]
delta_px = float((d_max - d_min) / max(bins-1, 1)) # ESFサンプル間隔 [pixel]

# --- LSF（SG微分） ---
LSF = savgol_filter(ESF, window_length=sg_win, polyorder=sg_poly, deriv=1, delta=delta_px)
# 面積を1に（数値安定化）
area = float(np.sum(LSF)*delta_px)
if abs(area) > 1e-12:
    LSF /= area
# --- FFT → MTF ---
win = get_window("hamming", len(LSF))
LSF_w = LSF * win
MTF = np.abs(rfft(LSF_w))
MTF /= (MTF[0] + 1e-12)
OTF=np.real(rfft(LSF_w))
OTF/=(OTF[0]+1e-12)

f_cyc_per_pix = rfftfreq(len(LSF_w), d=delta_px)  # [cycles/pixel]
f_lpmm_sensor = f_cyc_per_pix / pixel_pitch_mm    # [lp/mm]（センサ面）
x_ifft=np.fft.fftfreq(len(LSF_w),d=f_lpmm_sensor[2]-f_lpmm_sensor[1])


# MTF プロット
plt.figure(figsize=(7,4))
plt.plot(f_lpmm_sensor, MTF)
#plt.plot(f_obj, MTF)
plt.ylabel("MTF")
plt.title("MTF")
plt.xlim(0, 23.53)
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()

# 1) 最大 index を求める
idx_max = np.argmax(LSF_w)

# 2) ピクセル座標を mm に変換 【修正箇所】
Nx_lsf = len(LSF_w)

# 1要素あたりの物理距離 [mm]
# delta_px [pixel] * pixel_pitch_mm [mm/pixel]
step_mm = delta_px * pixel_pitch_mm 
shift=0#idx_max
# 中心をピークに合わせて座標配列を作成
x_mm = (np.arange(Nx_lsf) - idx_max+shift) * step_mm

# LSF プロット
plt.figure(figsize=(7,4))
plt.plot(x_mm, LSF_w / np.max(LSF_w), label="LSF")
plt.xlabel("Position [mm] (peak at 0 mm)")
plt.ylabel("LSF (normalized)")
plt.title("Line Spread Function")
# 表示範囲を適切に（例: ±0.1mm）
plt.xlim(-0.2, 0.2) 
plt.grid(True)
plt.tight_layout()

#ESF プロット
plt.figure(figsize=(7,4))
plt.plot(x_mm, ESF)
#plt.plot(f_obj, MTF)
plt.ylabel("ESF")
plt.title("ESF")
plt.xlim(-0.2, 0.2)
plt.ylim(0, 1.05)
plt.grid(True)
plt.tight_layout()

# --- 可視化 ---
plt.figure(figsize=(13,4.8))
plt.subplot(1,3,1)
overlay = cv2.cvtColor((roi/np.max(roi+1e-6)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
edge_mask = np.abs(d) < 0.75
overlay[edge_mask] = (0,255,0)
plt.imshow(overlay[...,::-1]); plt.title("ROI & 指定エッジ(緑)"); plt.axis('off')

plt.subplot(1,3,2)
plt.plot(x_mm, ESF); plt.grid(True); plt.xlabel("Distance [pixel]"); plt.ylabel("ESF (0-1)")
plt.title("ESF (binned, oversample=%d)"%oversample)

plt.subplot(1,3,3)
plt.plot(f_lpmm_sensor, MTF)
plt.xlabel("Spatial Frequency [lp/mm] (sensor)"); plt.ylabel("MTF"); plt.grid(True)
plt.title("MTF (Nyquist= %.1f lp/mm)"%(0.5/pixel_pitch_mm))
plt.xlim(0, 24); plt.ylim(0, 1.05)
plt.tight_layout(); plt.show()

print(f"角度θ(固定) = {theta_deg:.3f} deg,  指定点(ROI内) = ({xe:.1f}, {ye:.1f})")
# MTF50の目安
i50 = np.argmin(np.abs(MTF - 0.5))
print(f"MTF50 ≈ {f_lpmm_sensor[i50]:.2f} lp/mm (sensor)")

