import numpy as np
import matplotlib.pyplot as plt

illumi=True
mode=0 # 0: 点光源, 1: 画像
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

from PIL import Image  # 追加
if mode==1:
    image_path = "/home/yagilab/Downloads/white.png" # ← ここに読み込むPNGファイルのパスを指定

    # グレースケールで読み込み（0..255）→ 0..1 に正規化
    im = Image.open(image_path).convert('L')
    arr = np.array(im, dtype=np.float32) / 255.0  # shape: (Ny, Nx)
    arr = np.flipud(arr)

    # 画像サイズで Nx, Ny を決定
    Ny, Nx = arr.shape
    print("Size of object",Ny,Nx)
elif mode==0:
    Nx=64
    Ny=64



dpi=1200
inch=25.4
dx = inch/dpi  # [mm/px]
dy = inch/dpi  # [mm/px]
p_mm  = 25.4 / dpi      # 画素ピッチ [mm/pixel] （x,y 同一と仮定）
x_edges = (np.linspace(-Nx/2,Nx/2-1,Nx)) * dx  # [mm]
y_edges = (np.linspace(-Ny/2,Ny/2-1,Ny)) * dy  # [mm]
X,Y = np.meshgrid(x_edges,y_edges)
Eo=0
Ei=0




if mode==0:
# 配列は (Ny, Nx) にするのが慣例（[y, x]）
    arr = np.zeros((Ny, Nx), dtype=float)
    for i in range(Ny):
        for j in range(Nx):
            if ( ( (X[i,j])**2 + (Y[i,j])**2 ) <= (0.1)**2 ):
                arr[i][j]=1
# 配列は (Ny, Nx) にするのが慣例（[y, x]）
Image_map = np.zeros((Ny, Nx), dtype=float)

Li = 31.85
Lo = 30.7
Deg = 5/180*np.pi
Fnum=5.9
wavelen=570e-6

GRINcenterx = 0
GRINcentery = 0
dzo=0
zo=Lo+dzo
zi=Li

print("Zo",zo)


alpha=0.030041819731820713
n1=1.8999999999998573
L=35.13
d1=zo
R=1.1/2 #瞳半径

# alpha=0.1783
# n1=1.608
# L=20.4
# R=0.5225
ABCDmat=[[np.cos(alpha*L),np.sin(alpha*L)/(n1*alpha)],[-n1*alpha*np.sin(alpha*L),np.cos(alpha*L)]]
A=ABCDmat[0][0]
B=ABCDmat[0][1]
C=ABCDmat[1][0]
D=ABCDmat[1][1]
d2ideal=-(A*d1+B)/(C*d1+D)
d2=-(A*Lo+B)/(C*Lo+D)
print(d2)
dz=d2-d2ideal
M=-1/(C*d1+D)

# x0=(np.linspace(-Nx/2,Nx/2-1,Nx)) * dx
# y0=(np.linspace(-Ny/2,Ny/2-1,Ny)) * dy

# vector=[x0,y0,d1]
# norm=np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
# a0,b0,c0=vector[0]/norm,vector[1]/norm,vector[2]/norm



deltaz=1e-2
v_cutoff=2*R / (wavelen * d2ideal)
dfx=1/(Nx*dx)
dfy=1/(Ny*dy)
Npad=int((v_cutoff*1.1)/dfx)

fx2=(np.linspace(-Nx/2-Npad,Nx/2-1+Npad,Nx+2*Npad)) * dfx
fy2=(np.linspace(-Ny/2-Npad,Ny/2-1+Npad,Ny+2*Npad)) * dfy
FX2,FY2=np.meshgrid(fx2,fy2)
print("fx2max",fx2[-1])
# --- サンプリング ---
x0_1d = np.linspace(-R, R, Nx)
y0_1d = np.linspace(-R, R, Ny)
X0, Y0 = np.meshgrid(fx2*wavelen*d2ideal, fy2*wavelen*d2ideal)   # shape = (Ny, Nx)

# --- 初期方向余弦（光軸方向からの傾き） ---
norm = np.sqrt(X0**2 + Y0**2 + d1**2)
A0 = X0 / norm     # x方向の方向余弦
B0 = Y0 / norm     # y方向の方向余弦
C0 = d1 / norm     # z方向の方向余弦（≈1に近い）

# --- z軸配列 ---
z = np.arange(0, L + deltaz, deltaz)
Z = z[:, np.newaxis, np.newaxis]     # shape = (Nz,1,1)

# --- 定数項 ---
den = np.sqrt(n1**2 - n1**2*alpha**2*(X0**2 + Y0**2) - (1 - C0**2))

# --- 光線軌跡 ---
XZ = np.cos(n1*alpha*Z/den)*X0 + (1/(n1*alpha))*np.sin(n1*alpha*Z/den)*A0
YZ = np.cos(n1*alpha*Z/den)*Y0 + (1/(n1*alpha))*np.sin(n1*alpha*Z/den)*B0

# --- 屈折率分布と積分 ---
N2 = n1**2 * (1 - alpha**2 * ((XZ)**2 + (YZ)**2))
OPL_num = np.trapezoid(N2, z, axis=0)        # 台形則で z 方向に積分
OPL_den = np.sqrt(n1**2 - n1**2*alpha**2*(X0**2+Y0**2) - (1 - C0**2))
OPL = OPL_num / OPL_den #+np.sqrt(d1**2+X0**2+Y0**2)                  # shape = (Ny, Nx)
# z = np.arange(0, L + deltaz, deltaz)
# x0 = x0[np.newaxis, :]      # shape (1, Nx)
# y0 = y0[np.newaxis, :]
# z  = z[:, np.newaxis]       # shape (Nz, 1)
# xz=np.cos(n1*alpha*z/np.sqrt(n1*n1-n1*n1*alpha*alpha*(x0**2+y0**2)-(1-c0**2)))*x0+1/(n1*alpha)*np.sin(n1*alpha*z/np.sqrt(n1*n1-n1*n1*alpha*alpha*(x0**2+y0**2)-(1-c0**2)))*a0
# yz=np.cos(n1*alpha*z/np.sqrt(n1*n1-n1*n1*alpha*alpha*(x0**2+y0**2)-(1-c0**2)))*y0+1/(n1*alpha)*np.sin(n1*alpha*z/np.sqrt(n1*n1-n1*n1*alpha*alpha*(x0**2+y0**2)-(1-c0**2)))*b0

# n2=n1**2*(1-alpha**2*(xz**2+yz**2))
# OPL_d=np.sqrt(n1**2-n1**2*alpha**2*(x0**2+y0**2)-(1-c0**2))
# OPL_n=0
# for i in range(len(n2)-1):
#     if (xz[i]**2+yz[i]**2)>R or (xz[i+1]**2+yz[i+1]**2)>R:
#         OPL_n=0
#         break
#     OPL_n+=(n2[i]+n2[i+1])*deltaz/2

# OPL=OPL_n/OPL_d


k=2*np.pi/wavelen
delta=OPL+np.sqrt((d2)**2+(FX2*wavelen*d2ideal)**2+(FY2*wavelen*d2ideal)**2)-d1-n1*L-(d2)
P3_aperture = np.where((FX2*wavelen*d2ideal)**2 + (FY2*wavelen*d2ideal)**2 <= (R)**2, 1.0, 0.0)
Ws=((np.sqrt((FX2*wavelen*d2ideal)**2+(FY2*wavelen*d2ideal)**2))**4)
b=2000e-6
W=delta+b*Ws
print("W",W)
T3=P3_aperture*np.exp(-1j*k*W)




h=M*np.exp(-1j*k*(d1+n1*L+d2))/(wavelen**2*d2**2)*np.exp(-1j*k*((X0)**2+(Y0)**2)/(2*d2))*np.fft.ifft2(T3)
#h=np.fft.ifft2(np.fft.ifftshift(T3))
PSF=abs(h)**2
OTF=np.fft.fftshift(np.fft.fft2(PSF))
OTF=OTF[Npad:-Npad, Npad:-Npad]
OTF=OTF/np.max(OTF)
MTF=np.abs(OTF)
MTF=MTF/np.max(MTF)
Image_map=arr
Image_map=rescale_by_magnification(Image_map, dx, dy, 1/M)

# ----------------------------
# 入力：無収差の像（例）
# ----------------------------
# あなたの像配列（2D, float）。ここでは例として中央に点を置く。
# 既に img があるなら、このブロックは不要。
img = Image_map
#img[Ny//2, Nx//2] = 1.0

# ----------------------------
# センサー条件（1200 dpi 例）
# ----------------------------

# もし dx, dy を使っているなら p_mm = dx = dy に置き換えてOK

fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=p_mm))
fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=p_mm))
FX, FY = np.meshgrid(fx, fy)
fr = np.sqrt(FX**2 + FY**2)  # 放射周波数


Fimg  = np.fft.fftshift(np.fft.fft2(img))
Fout  = Fimg * OTF
img_a = (np.fft.ifft2(np.fft.ifftshift(Fout)))
img_b=np.abs(img_a)
# ----------------------------
# 可視化
# ----------------------------
# MTF（放射方向に一列を抜いて表示）
center_row = Ny//2
plt.figure(figsize=(7,4))
plt.plot(fr[center_row,:], MTF[center_row,:], label='Diffraction-limited',color='blue')
plt.axvline(v_cutoff, color='r', ls='--', lw=1, label=f'Cutoff={v_cutoff:.1f} cy/mm')
plt.xlim(0, fr[center_row,-1])
plt.ylim(0, 1.05)
plt.xlabel('Spatial frequency [cycles/mm]')
plt.ylabel('MTF')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# 像（前後比較）
# ==== ここを差し替え：像（前後比較）を mm 軸つきで表示 ====
# ピクセル境界（エッジ）を mm 単位で作る
# x_edges_mm = (np.arange(Nx + 1) - Nx/2) * dx
# y_edges_mm = (np.arange(Ny + 1) - Ny/2) * dy

# vmax = max(arr.max(), img_b.max())

# fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# im0 = axes[0].imshow(
#     arr,
#     extent=[x_edges_mm[0], x_edges_mm[-1], y_edges_mm[0], y_edges_mm[-1]],
#     origin='lower', cmap='hot', vmin=0, vmax=vmax, aspect='equal'
# )
# axes[0].set_title('Input (aberration-free)')
# axes[0].set_xlabel('x [mm]')     # ← 単位ラベル
# axes[0].set_ylabel('y [mm]')
# axes[0].axhline(0, color='w', lw=0.6, alpha=0.4)
# axes[0].axvline(0, color='w', lw=0.6, alpha=0.4)
# cbar0 = plt.colorbar(im0, ax=axes[0])
# cbar0.set_label('Intensity [a.u.]')

# im1 = axes[1].imshow(
#     img_b,
#     extent=[x_edges_mm[0], x_edges_mm[-1], y_edges_mm[0], y_edges_mm[-1]],
#     origin='lower', cmap='hot', vmin=0, vmax=vmax, aspect='equal'
# )
# axes[1].set_title('After MTF (diffraction × Gaussian)')
# axes[1].set_xlabel('x [mm]')     # ← 単位ラベル
# axes[1].set_ylabel('y [mm]')
# axes[1].axhline(0, color='w', lw=0.6, alpha=0.4)
# axes[1].axvline(0, color='w', lw=0.6, alpha=0.4)
# cbar1 = plt.colorbar(im1, ax=axes[1])
# cbar1.set_label('Intensity [a.u.]')

# plt.tight_layout()
plt.show()

from PIL import Image
import os

# # 出力フォルダとファイル名を設定
# save_dir = "/home/yagilab/Desktop/research_data/output_image"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "GRINlens_sim_result.tiff")

# # 正規化（0～1 → 0～255 へ）
# img_out = np.clip(img_b / np.max(img_b), 0, 1)
# img_uint8 = (img_out * 255).astype(np.uint8)

# # TIFFとして保存
# Image.fromarray(img_uint8).save(save_path, format='TIFF')

