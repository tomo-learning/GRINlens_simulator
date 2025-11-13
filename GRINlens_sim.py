import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
Nx=64             # x方向のサンプリング数
Ny=64             # y方向のサンプリング数
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

#===パラメータ=================================================
Lo = 30.7 #infocus時の物体-レンズ間距離
wavelen=570e-6
k=2*np.pi/wavelen
dzo=-5 #デフォーカス量
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


v_cutoff=2*R / (wavelen * d2ideal) #カットオフ周波数

#--カットオフ周波数がナイキスト周波数より高いためカットオフ周波数の1.1倍まで広げる----
dfx=1/(Nx*dx)
dfy=1/(Ny*dy)
Npad=int((v_cutoff*1.1)/dfx)
fx2=(np.linspace(-Nx/2-Npad,Nx/2-1+Npad,Nx+2*Npad)) * dfx
fy2=(np.linspace(-Ny/2-Npad,Ny/2-1+Npad,Ny+2*Npad)) * dfy
FX2,FY2=np.meshgrid(fx2,fy2)
NT=len(fx2)

#--空間領域の軸を作り直す------------------
dx2=1/(dfx*NT)
dy2=1/(dfy*NT)
x20=(np.linspace(-NT/2,NT/2-1,NT)) * dx2
y20=(np.linspace(-NT/2,NT/2-1,NT)) * dy2
X2,Y2=np.meshgrid(x20,y20)
#--------------------------------------------


# --- 初期方向余弦（光軸方向からの傾き） ---
norm = np.sqrt((FX2*wavelen*d2ideal)**2 + (FY2*wavelen*d2ideal)**2 + d1**2)
A0 = (FX2*wavelen*d2ideal) / norm     # x方向の方向余弦
B0 = (FY2*wavelen*d2ideal) / norm     # y方向の方向余弦
C0 = d1 / norm     # z方向の方向余弦

# --- z軸配列 ---
deltaz=1e-2
z = np.arange(0, L + deltaz, deltaz)
Z = z[:, np.newaxis, np.newaxis]     # shape = (Nz,1,1)

# --- 光線軌跡 式(3a) ---
den = np.sqrt(n1**2 - n1**2*alpha**2*((FX2*wavelen*d2ideal)**2 + (FY2*wavelen*d2ideal)**2) - (1 - C0**2))
XZ = np.cos(n1*alpha*Z/den)*(FX2*wavelen*d2ideal) + (1/(n1*alpha))*np.sin(n1*alpha*Z/den)*A0
YZ = np.cos(n1*alpha*Z/den)*(FY2*wavelen*d2ideal) + (1/(n1*alpha))*np.sin(n1*alpha*Z/den)*B0

# --- 屈折率分布と積分 式(8) ---
N2 = n1**2 * (1 - alpha**2 * ((XZ)**2 + (YZ)**2))
OPL_num = np.trapz(N2, z, axis=0)        # 台形則で z 方向に積分
OPL_den = np.sqrt(n1**2 - n1**2*alpha**2*((FX2*wavelen*d2ideal)**2+(FY2*wavelen*d2ideal)**2) - (1 - C0**2))
OPL = OPL_num / OPL_den +np.sqrt(d1**2+(FX2*wavelen*d2ideal)**2+(FY2*wavelen*d2ideal)**2) 


# ---- 収差関数 OPD 式(16)
delta=OPL+np.sqrt((d2)**2+(FX2*wavelen*d2ideal)**2+(FY2*wavelen*d2ideal)**2)-d1-n1*L-(d2ideal)

# ----収差伝達関数 式(18)
P3_aperture = np.where((FX2*wavelen*d2ideal)**2 + (FY2*wavelen*d2ideal)**2 <= (R)**2, 1.0, 0.0)
T3=P3_aperture*np.exp(-1j*k*delta)

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
plt.xlim(0, 70)
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


