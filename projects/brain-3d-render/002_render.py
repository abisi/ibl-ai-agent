"""002 -- GPU volume rendering of both channels (VTK ray casting, independent components: each channel its own colour and
opacity ramp, blended in one volume). Off-screen EGL on a Kuma GPU node.
Views: dorsal, lateral (left), oblique, oblique with a coronal cut through the DiI tracks; 360 deg rotation movie.
Display ranges: tissue percentiles per channel (config `render`). Axes are reordered to x = ML (right positive),
y = AP (anterior positive), z = DV (dorsal positive) from the fused "asr" volume (origin anterior, superior, right).
Usage: python 002_render.py [--factor 4] [--no-movie] [--tag quick]
Output: <work>/<brain>/render/<brain>_<view>[_tag].png, <brain>_rotation[_tag].mp4, render_info.json
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")

import numpy as np  # noqa: E402
import pyvista as pv  # noqa: E402
import vtk  # noqa: E402
import yaml  # noqa: E402

cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))
work = Path(cfg["paths"]["work"]) / cfg["brain"]


def load(factor):
    vols = []
    for ch in cfg["channels"]:
        v = np.load(work / f"ds{factor}_{ch['name']}.npy", mmap_mode="r")
        v = np.asarray(v).transpose(2, 0, 1)[::-1, ::-1, ::-1]   # (ML right->left, AP, DV) -> x right, y ant, z dors
        vols.append(v)
    return vols


def tissue_mask(af):
    sub = af[::4, ::4, ::4].astype(np.float32)
    bg, top = np.percentile(sub, 5), np.percentile(sub, 99)
    return af > bg + 0.15 * (top - bg)


def to_uint8(v, mask, lo_pct, hi_pct):
    t = v[mask][::50].astype(np.float32)
    lo, hi = np.percentile(t, lo_pct), np.percentile(t, hi_pct)
    out = np.clip((v.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8), (float(lo), float(hi))


def build_volume(vols, spacing):
    rc = cfg["render"]
    mask = tissue_mask(vols[0])
    comps, ranges = [], {}
    for ch, v in zip(cfg["channels"], vols):
        r = rc[ch["name"]]
        u8, ranges[ch["name"]] = to_uint8(v, mask, r["lo_pct"], r["hi_pct"])
        if ch["name"] != "autofluorescence":
            u8[~mask] = 0          # no DiI glow outside the brain (agarose / edge reflections)
        comps.append(u8)
    grid = pv.ImageData(dimensions=vols[0].shape, spacing=(spacing,) * 3)
    grid.point_data["v"] = np.stack([c.ravel(order="F") for c in comps], axis=1)
    prop = vtk.vtkVolumeProperty()
    prop.SetIndependentComponents(True)
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOff()
    for i, ch in enumerate(cfg["channels"]):
        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(0, 0, 0, 0)
        ctf.AddRGBPoint(255, *ch["color"])
        otf = vtk.vtkPiecewiseFunction()
        for x, a in rc[ch["name"]]["opacity"]:
            otf.AddPoint(255 * x, a)
        prop.SetColor(i, ctf)
        prop.SetScalarOpacity(i, otf)
        prop.SetComponentWeight(i, 1.0)
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(grid)
    mapper.SetBlendModeToComposite()
    mapper.SetSampleDistance(spacing * 0.5)
    mapper.AutoAdjustSampleDistancesOff()
    vol = vtk.vtkVolume()
    vol.SetMapper(mapper)
    vol.SetProperty(prop)
    dii = comps[1] > 128
    track_ap = float(np.median(np.nonzero(dii)[1]) * spacing) if dii.any() else grid.center[1]
    return grid, vol, mapper, ranges, track_ap


def set_view(pl, grid, azimuth, elevation, zoom=1.35):
    c = np.array(grid.center)
    r = np.linalg.norm(np.array(grid.bounds[1::2]) - np.array(grid.bounds[::2]))
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    # azimuth 0 = camera on the left side (-x), 90 = in front (+y, anterior); elevation from the horizontal plane
    d = np.array([-np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pl.camera.focal_point = c
    pl.camera.position = c + 2.2 * r * d
    pl.camera.up = (0, 1, 0) if elevation > 85 else (0, 0, 1)
    pl.camera.zoom(zoom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=cfg["render"]["factor"])
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = work / "render"
    out.mkdir(parents=True, exist_ok=True)
    sfx = f"_{a.tag}" if a.tag else ""
    spacing = cfg["voxel_um"] * a.factor
    grid, vol, mapper, ranges, track_ap = build_volume(load(a.factor), spacing)
    pv.global_theme.background = cfg["render"]["background"]
    pl = pv.Plotter(off_screen=True, window_size=cfg["render"]["window"])
    pl.add_actor(vol)
    views = {"dorsal": (0, 90), "lateral_left": (0, 0), "oblique": (35, 28), "oblique_back": (215, 28)}
    for name, (az, el) in views.items():
        set_view(pl, grid, az, el)
        pl.screenshot(out / f"{cfg['brain']}_{name}{sfx}.png")
        print("saved", name, flush=True)
    # coronal cut at the DiI tracks: anterior part removed, cut face toward the camera
    plane = vtk.vtkPlane()
    plane.SetOrigin(grid.center[0], track_ap, grid.center[2])
    plane.SetNormal(0, -1, 0)
    mapper.AddClippingPlane(plane)
    set_view(pl, grid, 60, 25)
    pl.screenshot(out / f"{cfg['brain']}_cut_tracks{sfx}.png")
    mapper.RemoveAllClippingPlanes()
    if not a.no_movie:
        m = cfg["render"]["movie"]
        pl.open_movie(str(out / f"{cfg['brain']}_rotation{sfx}.mp4"), framerate=m["fps"], quality=8)
        for k in range(m["n_frames"]):
            set_view(pl, grid, 360 * k / m["n_frames"], 22)
            pl.write_frame()
        pl.close()
        print("saved movie", flush=True)
    json.dump({"factor": a.factor, "spacing_um": spacing, "display_ranges_raw": ranges, "track_ap_um": track_ap,
               "views": views, "vtk": vtk.vtkVersion.GetVTKVersion(), "pyvista": pv.__version__},
              open(out / f"render_info{sfx}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
