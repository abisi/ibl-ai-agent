"""Contact sheet of all 001 example rasters (multi-page PDF, 6 per page), sessions ordered by the combined rank of
unit count and population response per unit, then by example rank within the session."""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

OUT = axel_bisi_path('combined_results_ks4', 'ssl-session-raster-examples')
d = pd.read_csv(OUT / 'tables' / '001_example_ranking.csv')
best = d[d['rank'] == 1].copy()
best['combo'] = best['n_units'].rank(pct=True) + best['score_per_unit'].rank(pct=True)
order = best.sort_values(['preferred', 'combo'], ascending=False)['session_id'].tolist()
d['session_order'] = d['session_id'].map({s: i for i, s in enumerate(order)})
d = d.sort_values(['session_order', 'rank'])

W, H, NCOL, NROW = 1100, 970, 3, 2
font = ImageFont.load_default(size=26)
pages, page = [], None
for k, row in enumerate(d.itertuples()):
    if k % (NCOL * NROW) == 0:
        page = Image.new('RGB', (W * NCOL, (H + 50) * NROW), 'white')
        pages.append(page)
    im = Image.open(row.figure).convert('RGB')
    im.thumbnail((W, H))
    i = k % (NCOL * NROW)
    x, y = (i % NCOL) * W, (i // NCOL) * (H + 50)
    page.paste(im, (x, y + 50))
    flag = '' if row.preferred else '  (no trial passing all criteria)'
    ImageDraw.Draw(page).text(
        (x + 20, y + 10), f'{row.session_id}  day {int(row.day)}  #{row.rank}  t={row.t_stim:.0f}s  '
        f'{int(row.n_units)} units  jaw {1e3 * row.rt_jaw:.0f} / lick {1e3 * row.rt_lick:.0f} ms{flag}',
        fill='black', font=font)
pages[0].save(OUT / 'example_contact_sheet.pdf', save_all=True, append_images=pages[1:], resolution=150)
pages[0].save(OUT / 'example_contact_sheet_page1.png')
print(len(d), 'examples on', len(pages), 'pages ->', OUT / 'example_contact_sheet.pdf')
