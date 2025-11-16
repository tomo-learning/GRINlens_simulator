import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata

#像倍率Mでリサイズする関数
def rescale_by_magnification(img_b, dx, dy, M, alpha=None, fill=0.0):
    """
    I'(x,y) = alpha * I(Mx, My) を実装（双一次補間）。
    img_b: 2D 強度画像 (Ny, Nx)
    dx, dy: [mm/px]
    M: zideal/zo
    alpha: 係数（既定はエネルギー保存の M**2）
    fill: 範囲外画素の値
    """
    Ny, Nx = img_b.shape
    if alpha is None:
        alpha = M**2  # 強度のエネルギー（積分）保存を優先

    # 出力画像の座標（中心を原点に）
    x = (np.arange(Nx) - Nx/2) * dx   # [mm]
    y = (np.arange(Ny) - Ny/2) * dy   # [mm]
    X, Y = np.meshgrid(x, y)          # 形状 (Ny, Nx)

    # 入力画像上のサンプリング位置（mm座標を M 倍）
    U = M * X
    V = M * Y

    # mm -> ピクセル添字（0..Nx-1, 0..Ny-1）へ変換
    u = U / dx + Nx/2
    v = V / dy + Ny/2

    # 双一次補間のための周辺画素添字
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    # 重み
    du = u - u0
    dv = v - v0

    # 範囲マスク
    mask00 = (u0 >= 0) & (u0 < Nx) & (v0 >= 0) & (v0 < Ny)
    mask10 = (u1 >= 0) & (u1 < Nx) & (v0 >= 0) & (v0 < Ny)
    mask01 = (u0 >= 0) & (u0 < Nx) & (v1 >= 0) & (v1 < Ny)
    mask11 = (u1 >= 0) & (u1 < Nx) & (v1 >= 0) & (v1 < Ny)

    out = np.full((Ny, Nx), fill, dtype=img_b.dtype)

    # 取り出し（範囲外は無視）
    I00 = np.zeros_like(out); I10 = np.zeros_like(out)
    I01 = np.zeros_like(out); I11 = np.zeros_like(out)

    I00[mask00] = img_b[v0[mask00], u0[mask00]]
    I10[mask10] = img_b[v0[mask10], u1[mask10]]
    I01[mask01] = img_b[v1[mask01], u0[mask01]]
    I11[mask11] = img_b[v1[mask11], u1[mask11]]

    # 双一次補間
    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv

    out = alpha * (w00 * I00 + w10 * I10 + w01 * I01 + w11 * I11)
    return out

#--センサ面のパラメータ--
Nx=128            # x方向のサンプリング数
Ny=128            # y方向のサンプリング数
dpi=1200          # センサのdpi
inch=25.4         # インチ[mm]
dx = inch/dpi     # x方向のサンプリング間隔 [mm/px]
dy = inch/dpi     # y方向のサンプリング間隔 [mm/px]

#--センサ面の格子------------------------------
x = (np.linspace(-Nx/2,Nx/2-1,Nx)) * dx  # [mm]
y = (np.linspace(-Ny/2,Ny/2-1,Ny)) * dy  # [mm]
X,Y = np.meshgrid(x,y)
#----------------------------------------------

#--物体の定義-----------------------------------
Object_map = np.zeros((Ny, Nx), dtype=float)
for i in range(Ny):
    for j in range(Nx):
        if ( ( (X[i,j])**2 + (Y[i,j])**2 ) <= (0.1)**2 ):
            Object_map[i][j]=1
#-----------------------------------------------

#-- plane2----------------------------------------
Nx2=Nx
Ny2=Ny
dx2=dx
dy2=dy
x2=(np.linspace(-Nx2/2,Nx2/2-1,Nx2)) * dx2  # [mm]
y2=(np.linspace(-Ny2/2,Ny2/2-1,Ny2)) * dy2  # [mm]
X2,Y2 = np.meshgrid(x2,y2)




#===パラメータ=================================================
Lo = 30.7 #infocus時の物体-レンズ間距離
wavelen=570e-6
k=2*np.pi/wavelen
dzo=-1.5 #デフォーカス量
d1=Lo+dzo # デフォーカス時の物体-レンズ間距離

#--最小二乗法で求めたGRINレンズパラメータ
alpha=0.03148084603797629
n1=1.6531323196563283
#-------------------------------------

L=35.13 # GRINレンズのz方向の長さ
R=0.5 #瞳半径
#====================================================================

#-- 一次光学行列--------------------------------------------
A=np.cos(alpha*L)
B=np.sin(alpha*L)/(n1*alpha)
C=-n1*alpha*np.sin(alpha*L)
D=np.cos(alpha*L)
#--------------------------------------------------------

#--defocus時の理想結像距離をd2ideal、実際のレンズ-センサ間距離をd2としている (レンズ-センサ間の距離を固定にしているため)
d2ideal=-(A*d1+B)/(C*d1+D)
d2=-(A*Lo+B)/(C*Lo+D)
M=-1/(C*d1+D) #倍率


v_cutoff=2*R / (wavelen * d2) #カットオフ周波数

#--カットオフ周波数がナイキスト周波数より高いためカットオフ周波数の1.1倍まで広げる----

#--------------------------------------------


# --- 初期方向余弦（光軸方向からの傾き） ---
norm = np.sqrt((X2)**2 + (Y2)**2 + d1**2)
A0 = (X2) / norm     # x方向の方向余弦
B0 = (Y2) / norm     # y方向の方向余弦
C0 = d1 / norm     # z方向の方向余弦

# --- z軸配列 ---
deltaz=1e-2
z = np.arange(0, L + deltaz, deltaz)
Z = z[:, np.newaxis, np.newaxis]     # shape = (Nz,1,1)

# --- 光線軌跡 式(3a) ---
den = np.sqrt(n1**2 - n1**2*alpha**2*((X2)**2 + (Y2)**2) - (1 - C0**2))
XZ = np.cos(n1*alpha*Z/den)*(X2) + (1/(n1*alpha))*np.sin(n1*alpha*Z/den)*A0
YZ = np.cos(n1*alpha*Z/den)*(Y2) + (1/(n1*alpha))*np.sin(n1*alpha*Z/den)*B0

X3 = np.cos(n1*alpha*L/den)*(X2) + (1/(n1*alpha))*np.sin(n1*alpha*L/den)*(X2/norm)
Y3 = np.cos(n1*alpha*L/den)*(Y2) + (1/(n1*alpha))*np.sin(n1*alpha*L/den)*(Y2/norm)


# --- 屈折率分布と積分 式(8) ---
N2 = n1**2 * (1 - alpha**2 * ((XZ)**2 + (YZ)**2))
OPL_num = np.trapz(N2, z, axis=0)        # 台形則で z 方向に積分
OPL_den = np.sqrt(n1**2 - n1**2*alpha**2*((X2)**2+(X2)**2) - (1 - C0**2))

OPL = OPL_num / OPL_den +d1 #+np.sqrt(d1**2+(X2)**2+(Y2)**2) 


# ---- 収差関数 OPD 式(16)
delta=OPL+np.sqrt(float(d2ideal)**2+(X3)**2+(Y3)**2)-d1-n1*L-(d2)

# ----収差伝達関数 式(18)
P3_aperture = np.where((X3)**2 + (Y3)**2 <= (R)**2, 1.0, 0.0)
T3=P3_aperture*np.exp(-1j*k*delta)


# --- 目的の等間隔格子（FFTに使いやすい正方格子・2のべき推奨） ---
xg = x.copy()   # すでに dx 間隔
yg = y.copy()   # すでに dy 間隔
Xg, Yg = np.meshgrid(xg, yg)

# --- 入力散布点（X3,Y3 はあなたの計算結果; 単位[mm]） ---
pts = np.column_stack([X3.ravel(), Y3.ravel()])

# --- 複素場を実部・虚部で補間 ---
vals_re = np.real(T3).ravel()
vals_im = np.imag(T3).ravel()

Tg_re = griddata(pts, vals_re, (Xg, Yg), method='linear')
Tg_im = griddata(pts, vals_im, (Xg, Yg), method='linear')

# 線形補間で埋まらない凸包外は最近傍で補完
Tg_re_nn = griddata(pts, vals_re, (Xg, Yg), method='nearest')
Tg_im_nn = griddata(pts, vals_im, (Xg, Yg), method='nearest')

Tg_re = np.where(np.isnan(Tg_re), Tg_re_nn, Tg_re)
Tg_im = np.where(np.isnan(Tg_im), Tg_im_nn, Tg_im)
Tg = Tg_re + 1j*Tg_im

# --- 開口外を0に（安全策） ---
apert = (Xg**2 + Yg**2) <= R**2
Tg *= apert.astype(Tg.dtype)

# # ビンの境界（エッジ）を dx/2, dy/2 だけ外側にずらしたものとして作る
# x_edges = np.concatenate([xg - dx/2, [xg[-1] + dx/2]])
# y_edges = np.concatenate([yg - dy/2, [yg[-1] + dy/2]])

# # 実部・虚部を座標に応じて 2Dヒストグラム集計
# vals_re = np.real(T3).ravel()
# vals_im = np.imag(T3).ravel()

# H_re, _, _ = np.histogram2d(
#     Y3.ravel(), X3.ravel(),
#     bins=[y_edges, x_edges],
#     weights=vals_re
# )
# H_im, _, _ = np.histogram2d(
#     Y3.ravel(), X3.ravel(),
#     bins=[y_edges, x_edges],
#     weights=vals_im
# )
# Cnt,  _, _ = np.histogram2d(
#     Y3.ravel(), X3.ravel(),
#     bins=[y_edges, x_edges]
# )

# # ビン内の複素平均（Cnt=0 のところは 0 に）
# Tg = (H_re + 1j*H_im) / np.maximum(Cnt, 1)

# 開口外は 0 に
# apert = (Xg**2 + Yg**2) <= R**2
# Tg *= apert.astype(Tg.dtype)


# === 振幅ホットマップ ===
plt.figure(figsize=(6,5))
# plt.imshow(
#     np.abs(Tg),
#     extent=[xg[0], xg[-1], yg[0], yg[-1]],
#     cmap='hot', origin='lower'
# )
plt.imshow(
    np.abs(T3),
    extent=[X3[Ny//2,0], X3[Ny//2,-1], Y3[0,Nx//2], Y3[-1,Nx//2]],
    cmap='hot', origin='lower'
)
plt.colorbar(label='Amplitude')
plt.title('Pupil amplitude |Tg| (binned, dx,dy grid)')
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.tight_layout()

plt.figure(figsize=(7,4))
plt.plot(X2[Ny//2,:], X3[Ny//2,:],color='blue')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(7,4))
plt.plot(z, XZ[:,Ny//2,Nx//2+8], label='Ray trace',color='blue')
plt.xlim(0, z[-1])
plt.xlabel('z [mm]')
plt.ylabel('X(z) [mm]')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#--- PSF OTF MTF計算
#h=M*np.exp(-1j*k*(d1+n1*L+d2ideal))/(wavelen**2*d2ideal**2)*np.exp(-1j*k*((X2)**2+(Y2)**2)/(2*d2))*np.fft.ifft2(T3)*(wavelen**2*d2ideal**2)
h=np.fft.ifft2(np.fft.ifftshift(T3))
PSF=np.fft.fftshift(abs(h)**2)
OTF=np.fft.fftshift(np.fft.fft2(PSF))
OTF=OTF/np.max(OTF)
OTF_reshape=OTF[Npad:-Npad, Npad:-Npad] #センサ面と軸を合わせるためにクリッピング
MTF=np.abs(OTF)
MTF=MTF/np.max(MTF)


#-- 物体を倍率Mでリサイズ→OTFで畳み込み
Image_map=rescale_by_magnification(Object_map, dx, dy, 1/M)
Fimg  = np.fft.fftshift(np.fft.fft2(Image_map))
Fout  = Fimg * OTF_reshape
img_b = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Fout)))
img_b=np.abs(img_b)


# ----------------------------
# 可視化
# ----------------------------
# MTF（FY=0[lp/mm]で切り出し）
center_row = NT//2
plt.figure(figsize=(7,4))
plt.plot(FX2[center_row,:], MTF[center_row,:],color='blue')
plt.axvline(v_cutoff, color='r', ls='--', lw=1, label=f'Cutoff={v_cutoff:.1f} cy/mm')
plt.xlim(0, 23.5)
plt.ylim(0, 1.05)
plt.xlabel('Spatial frequency [cycles/mm]')
plt.ylabel('MTF')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()


# 像（前後比較）
# ==== ここを差し替え：像（前後比較）を mm 軸つきで表示 ====
# ピクセル境界（エッジ）を mm 単位で作る
x_edges_mm = (np.arange(Nx + 1) - Nx/2) * dx
y_edges_mm = (np.arange(Ny + 1) - Ny/2) * dy

vmax = max(Object_map.max(), img_b.max())

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

im0 = axes[0].imshow(
    Object_map,
    extent=[x_edges_mm[0], x_edges_mm[-1], y_edges_mm[0], y_edges_mm[-1]],
    origin='lower', cmap='hot', vmin=0, vmax=vmax, aspect='equal'
)
axes[0].set_title('Input (aberration-free)')
axes[0].set_xlabel('x [mm]')     # ← 単位ラベル
axes[0].set_ylabel('y [mm]')
axes[0].axhline(0, color='w', lw=0.6, alpha=0.4)
axes[0].axvline(0, color='w', lw=0.6, alpha=0.4)
cbar0 = plt.colorbar(im0, ax=axes[0])
cbar0.set_label('Intensity [a.u.]')

im1 = axes[1].imshow(
    img_b,
    extent=[x_edges_mm[0], x_edges_mm[-1], y_edges_mm[0], y_edges_mm[-1]],
    origin='lower', cmap='hot', vmin=0, vmax=vmax, aspect='equal'
)
axes[1].set_title('After MTF (diffraction × Gaussian)')
axes[1].set_xlabel('x [mm]')     # ← 単位ラベル
axes[1].set_ylabel('y [mm]')
axes[1].axhline(0, color='w', lw=0.6, alpha=0.4)
axes[1].axvline(0, color='w', lw=0.6, alpha=0.4)
cbar1 = plt.colorbar(im1, ax=axes[1])
cbar1.set_label('Intensity [a.u.]')

plt.tight_layout()
plt.show()


