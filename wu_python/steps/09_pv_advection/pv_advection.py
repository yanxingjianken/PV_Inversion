#!/usr/bin/env python3
"""Wu Python — Step 09: PV advection by piecewise-induced winds.

Uses Fortran-computed pv_advection.nc data. Generates pv_advection_all_pieces.png.
"""
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
from pathlib import Path; from scipy.ndimage import gaussian_filter; import sys
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from wu_python.core.grid import NY, NX, lats, lons, LON2D, LAT2D
STEP_DIR = Path(__file__).resolve().parent
PY_OUT = Path(_root) / "data" / "wu_python_out"
nc = PY_OUT / "pv_advection.nc"
if not nc.exists():
    nc = Path(_root) / "data" / "wu_out" / "pv_advection.nc"
    print(f"  Using Fortran NetCDF: {nc}")
ds = xr.open_dataset(nc)
q_event = ds.Q_event_250.values; q_anom = ds.Q_anom_250.values
# Compute gradients
R_E = 2.0e7/np.pi; DP=R_E*np.radians(1.5); DL=R_E*np.radians(1.5)
AP=np.cos(np.radians(lats))
q_fill=np.where(np.isnan(q_event), np.nanmean(q_event), q_event)
dqdx=np.zeros_like(q_fill); dqdy=np.zeros_like(q_fill)
dqdy[1:-1,:]=(q_fill[:-2,:]-q_fill[2:,:])/(2.*DP)
for i in range(1,NY-1): dqdx[i,1:-1]=(q_fill[i,2:]-q_fill[i,:-2])/(2.*DL*AP[i])
# Use saved Pass D from step 07 or Fortran equivalent
import config; WU_DIR=Path(config.WU_IN_DIR)
def read_wu_ascii(fp):
    d=[]; 
    with open(fp) as f:
        for l in f:
            for t in l.split(): d.append(float(t))
    return np.array(d[:8]),np.array(d[8:])
NW,block,NPIECES=10,NY*NX,3
_,vals=read_wu_ascii(WU_DIR/"event_pert.out")
n_per=2*NW*block
PSI=np.stack([vals[ip*n_per+(NW+7)*block:(ip*n_per+(NW+8)*block)].reshape(NY,NX) for ip in range(NPIECES)],axis=0)
U_ind=np.zeros((NPIECES,NY,NX)); V_ind=np.zeros((NPIECES,NY,NX))
for ip in range(NPIECES):
    for i in range(1,NY-1):
        U_ind[ip,i,1:-1]=-(PSI[ip,i-1,1:-1]-PSI[ip,i+1,1:-1])/(2.*DP)
        V_ind[ip,i,1:-1]=(PSI[ip,i,2:]-PSI[ip,i,:-2])/(2.*DL*AP[i])
# Cap winds at 40 m/s
for ip in range(NPIECES):
    spd=np.sqrt(U_ind[ip]**2+V_ind[ip]**2); mask=spd>40
    U_ind[ip][mask]*=40./spd[mask]; V_ind[ip][mask]*=40./spd[mask]
# PV advection
S=86400.; PVadv=np.zeros((NPIECES,NY,NX))
for ip in range(NPIECES):
    PVadv[ip]=-(U_ind[ip]*dqdx+V_ind[ip]*dqdy)*S
    PVadv[ip,0,:]=PVadv[ip,-1,:]=PVadv[ip,:,0]=PVadv[ip,:,-1]=np.nan
PVs=PVadv.copy(); PVs[2]=gaussian_filter(np.nan_to_num(PVadv[2]),sigma=1.5)
for ip in range(3): pv=PVadv[ip,1:-1,1:-1]; print(f"  Piece {ip+1}: {np.nanmin(pv):.1f} / {np.nanpercentile(pv,50):.1f} / {np.nanmax(pv):.1f} PVU/day")
# Plot
proj=ccrs.LambertConformal(central_longitude=-105,central_latitude=50); pc=ccrs.PlateCarree()
fig=plt.figure(figsize=(20,14))
for col,(data,title,cmap,vmx) in enumerate([
    (dqdx,"∂q/∂x [PVU/m]","RdBu_r",np.nanpercentile(np.abs(dqdx),98)),
    (dqdy,"∂q/∂y [PVU/m]","RdBu_r",np.nanpercentile(np.abs(dqdy),98))]):
    ax=fig.add_subplot(2,4,col+1,projection=proj)
    ax.set_extent([-175,-35,5,88],crs=pc); ax.add_feature(cfeature.COASTLINE,lw=0.3,edgecolor="0.5")
    cf=ax.pcolormesh(LON2D,LAT2D,data,cmap=cmap,transform=pc,vmin=-vmx,vmax=vmx)
    plt.colorbar(cf,ax=ax,shrink=0.7,pad=0.02); ax.set_title(title,fontsize=10,fontweight="bold")
for ip in range(3):
    ax=fig.add_subplot(2,4,ip+5,projection=proj)
    ax.set_extent([-175,-35,5,88],crs=pc); ax.add_feature(cfeature.COASTLINE,lw=0.4,edgecolor="0.4")
    vmx=np.nanpercentile(np.abs(PVs[ip]),98)
    cf=ax.pcolormesh(LON2D,LAT2D,PVs[ip],cmap="RdBu_r",transform=pc,vmin=-vmx,vmax=vmx)
    plt.colorbar(cf,ax=ax,shrink=0.7,pad=0.02,label="PVU/day")
    ax.set_title(f"{['(a) Lower','(b) Middle','(c) Upper [smoothed]'][ip]}\n[{np.nanmin(PVs[ip]):.0f},{np.nanmax(PVs[ip]):.0f}]",fontsize=10)
fig.delaxes(fig.add_subplot(2,4,4))
plt.suptitle("Wu Python — 250-hPa PV Advection\n2025-01-08 CA Blocking",fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig(STEP_DIR/"pv_advection_all_pieces.png",dpi=150,bbox_inches="tight"); plt.close()
print("  ✓ Saved: pv_advection_all_pieces.png")
ds.close()
print("✓ Step 09 complete → Step 10")
