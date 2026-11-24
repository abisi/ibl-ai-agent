"""Assemble the Figure 2 draft (vector, one 183-mm-wide page): A-D and E cropped from the user's layout Figure2_test.pdf
(layout scale kept), F = Figure2F_grid.pdf (F_roc_grid.py, 130 mm wide) beside E (user 2026-11-19), G =
G_response_maps.pdf (G_response_maps.py) below, at full width. Panel letters F, G added; the placeholder F of the layout
is masked.
Output: manuscript Figures/Figure2_sensory_maps_explore_2026-11-17/Figure2_draft.{pdf,png}
Run (haas): uvx --with pymupdf python -W ignore assemble_figure2.py [--ad-bottom --e-right --e-bottom FRAC]
"""
import argparse
from pathlib import Path

import fitz  # pymupdf

D = Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/Figure2_sensory_maps_explore_2026-11-17")
MM = 72 / 25.4
W = 183 * MM
GAP = 2 * MM
MAX_H = 247  # mm, Nature full page


def letter(page, x, y, s):
    page.insert_text((x + 0.3 * MM, y + 3.2 * MM), s, fontsize=8, fontname="hebo")


def main(a):
    test = fitz.open(D / "Figure2_test.pdf")
    r = test[0].rect
    sc = W / r.width

    def fr(x0, y0, x1, y1):
        return fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)

    ad, e = fr(0, 0, 1, a.ad_bottom), fr(0, a.ad_bottom, a.e_right, a.e_bottom)
    F, G = fitz.open(D / "F_roc" / "Figure2F_grid.pdf"), fitz.open(D / "G_maps" / "G_response_maps.pdf")
    h_ad, w_e, h_e = ad.height * sc, e.width * sc, e.height * sc
    w_f = F[0].rect.width                                              # F at its native size (130 mm)
    h_f = F[0].rect.height
    h_g = W * G[0].rect.height / G[0].rect.width
    h_mid = max(h_e, h_f)
    H = h_ad + h_mid + GAP + h_g
    out = fitz.open()
    page = out.new_page(width=W, height=H)
    page.show_pdf_page(fitz.Rect(0, 0, W, h_ad), test, 0, clip=ad)
    page.draw_rect(fitz.Rect(a.e_right * W, h_ad - 0.017 * r.height * sc, 0.45 * W, h_ad), color=None,
                   fill=(1, 1, 1), overlay=True)                       # placeholder letter F at the band bottom
    page.show_pdf_page(fitz.Rect(0, h_ad, w_e, h_ad + h_e), test, 0, clip=e)
    page.draw_rect(fitz.Rect(w_e, h_ad, W, h_ad + h_mid), color=None, fill=(1, 1, 1), overlay=True)
    x_f = W - w_f
    page.show_pdf_page(fitz.Rect(x_f, h_ad, W, h_ad + h_f), F, 0)
    letter(page, x_f - 3 * MM, h_ad - 1 * MM, "F")
    y_g = h_ad + h_mid + GAP
    page.show_pdf_page(fitz.Rect(0, y_g, W, y_g + h_g), G, 0)
    letter(page, 0, y_g, "G")
    out.save(D / "Figure2_draft.pdf")
    page.get_pixmap(dpi=200).save(D / "Figure2_draft.png")
    print(f"page 183 x {H / MM:.0f} mm (max {MAX_H}); A-D {h_ad / MM:.0f}, E {h_e / MM:.0f} ({w_e / MM:.0f} wide) | "
          f"F {h_f / MM:.0f} ({w_f / MM:.0f} wide), G {h_g / MM:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ad-bottom", type=float, default=0.175)
    ap.add_argument("--e-right", type=float, default=0.28)
    ap.add_argument("--e-bottom", type=float, default=0.465)
    main(ap.parse_args())
