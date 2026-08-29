# -*- coding: utf-8 -*-
"""
Figure: study area map (reviewer 1, item 1).

특별호 게재 3편의 Figure 1 구성(위치 + 지형 + 수문 관측소)에 맞춘다.
  - 지형 음영: GMRT(Global Multi-Resolution Topography) GridServer, resolution=med
    (캘리포니아 1008x1026, 칠레 1904x1596; data/hydro/elev_{tag}.npz 에 저장)
  - 별표: IVT 추출 ERA5 격자점(저장된 IVT 배열과 최대절대차 0.0000 으로 확인)
  - 원: 유량 검증에 사용한 USGS 관측소 4곳(캘리포니아, 격자점 40 km 이내)
  - 점선 원: 격자점 반경 40 km

Panel (a) California, (37.75N, 122.50W), IVT threshold 249.8 kg m-1 s-1
Panel (b) Central Chile, (33.00S, 71.50W), IVT threshold 161.8 kg m-1 s-1

Output: paper/fig_study_area.png (600 dpi), .pdf
Run: python make_fig_studyarea.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "paper")
os.makedirs(OUT_DIR, exist_ok=True)

# 유량 검증에 쓴 USGS 관측소 (site_no: 표시명, lat, lon)
GAUGES = [   # USGS site_no, 표시명, lat, lon, 격자점 거리 (sites.txt 원본 확인값)
    ("11181040", "San Lorenzo C", 37.6841, -122.1400, 32.5),
    ("11162630", "Pilarcitos C", 37.4666, -122.4341, 32.0),
    ("11460400", "Lagunitas C", 38.0268, -122.7363, 37.1),
    ("11181000", "San Lorenzo C (Hayward)", 37.6855, -122.0644, 39.0),
]

SITES = [
    dict(tag="ca", label="(a) California", pt=(37.75, -122.50), thr=249.8,
         name="San Francisco coast", extent=[-124.2, -120.4, 36.4, 39.0], gauges=True),
    dict(tag="chile", label="(b) Central Chile", pt=(-33.00, -71.50), thr=161.8,
         name="Valparaíso coast", extent=[-72.6, -68.8, -34.3, -31.7], gauges=False),
]


def load_elev(tag):
    p = os.path.join(HERE, "data", "hydro", f"elev_{tag}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["lat"], z["lon"], z["z"]


def ring(lat0, lon0, km=40, n=180):
    """중심에서 반경 km 인 원(위경도 근사)."""
    a = np.linspace(0, 2 * np.pi, n)
    dlat = km / 111.19
    dlon = km / (111.19 * np.cos(np.radians(lat0)))
    return lon0 + dlon * np.cos(a), lat0 + dlat * np.sin(a)


fig = plt.figure(figsize=(11.4, 5.4))
proj = ccrs.PlateCarree()

for i, s in enumerate(SITES):
    ax = fig.add_subplot(1, 2, i + 1, projection=proj)
    ax.set_extent(s["extent"], crs=proj)

    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#c9dced", zorder=0)

    el = load_elev(s["tag"])
    if el is not None:
        lat, lon, Z = el
        Zl = np.where(Z > 0, Z, np.nan)                  # 육지만
        ls = LightSource(azdeg=315, altdeg=45)
        # terrain 컬러맵의 앞 25%는 수심용 파랑이므로 육지에는 그 뒤 구간만 쓴다
        land_cmap = LinearSegmentedColormap.from_list(
            "land", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256)))
        norm = Normalize(vmin=0, vmax=float(np.nanpercentile(Zl, 99.5)))
        rgb = ls.shade(np.nan_to_num(Zl), cmap=land_cmap,
                       norm=norm, blend_mode="soft", vert_exag=25,
                       dx=1, dy=1)
        rgba = rgb[..., :3]                                  # shade 는 RGBA 반환
        rgba = np.dstack([rgba, np.where(np.isnan(Zl), 0.0, 1.0)])
        ax.imshow(rgba, origin="lower", transform=proj,
                  extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                  interpolation="bilinear", zorder=1)

    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.5,
                   edgecolor="#2b6ca3", zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7,
                   edgecolor="#333333", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4,
                   edgecolor="#666666", linestyle=":", zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#999999",
                      linestyle=":", alpha=0.6)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 8}

    lat0, lon0 = s["pt"]

    rx, ry = ring(lat0, lon0, 40)
    ax.plot(rx, ry, transform=proj, color="#c0392b", lw=0.9, ls="--",
            alpha=0.85, zorder=5)

    if s["gauges"]:
        for j, (sid, gn, gla, glo, gd) in enumerate(GAUGES):
            ax.plot(glo, gla, marker="o", ms=5.5, mfc="#f5f5f5", mec="#1a1a1a",
                    mew=0.9, transform=proj, zorder=6,
                    label="USGS streamflow gauge" if j == 0 else None)

    ax.plot(lon0, lat0, marker="*", ms=19, color="#c0392b",
            markeredgecolor="black", markeredgewidth=0.7, transform=proj,
            zorder=7, label="ERA5 grid point (IVT)")
    ax.annotate(s["name"], xy=(lon0, lat0), xytext=(13, 9),
                textcoords="offset points", fontsize=8.5, zorder=8,
                bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#888888",
                          lw=0.5, alpha=0.92))

    ax.set_title(f"{s['label']}   {abs(lat0):.2f}°{'N' if lat0 > 0 else 'S'}, "
                 f"{abs(lon0):.2f}°W\nIVT threshold {s['thr']:.1f} "
                 f"kg m$^{{-1}}$ s$^{{-1}}$", fontsize=10)

    if s["gauges"]:
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                  borderpad=0.4).set_zorder(9)

fig.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"fig_study_area.{ext}")
    fig.savefig(p, dpi=600, bbox_inches="tight")
    print("saved", p)
