"""Render the project report as a self-contained HTML page with the figures inlined."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
OUT = Path("/tmp/claude-1000/-home-ubuntu-project-wavefield-low-rank-representation/"
           "86678a6d-57f5-42c0-a3d9-45bf766bf8f3/scratchpad/delay_occupancy_report.html")


def figure(name: str) -> str:
    path = FIGURES / name
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


def summary() -> dict:
    path = ROOT / "reports" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


HEAD = """<title>延迟占据秩定律</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Spectral:wght@500;600&display=swap">
<style>
:root{
  --ground:#F4F6F8; --surface:#FFFFFF; --surface-sunk:#EEF1F5;
  --ink:#141B23; --ink-soft:#3D4954; --muted:#6A7683; --rule:#DDE3EA;
  --accent:#1F6FB2; --accent-soft:#E3EEF7;
  --negative:#B4413C; --negative-soft:#F7E7E5;
  --positive:#3F7A5E; --positive-soft:#E4F0EA;
  --shadow:0 1px 2px rgba(20,27,35,.06), 0 8px 24px -16px rgba(20,27,35,.28);
}
:root:not([data-theme="light"]){ }
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#101519; --surface:#171D23; --surface-sunk:#1D242B;
    --ink:#E7ECF1; --ink-soft:#BDC7D1; --muted:#8B96A2; --rule:#28313A;
    --accent:#63AAE6; --accent-soft:#16303F;
    --negative:#E2827B; --negative-soft:#33211F;
    --positive:#77B896; --positive-soft:#1B2C24;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }
}
:root[data-theme="dark"]{
  --ground:#101519; --surface:#171D23; --surface-sunk:#1D242B;
  --ink:#E7ECF1; --ink-soft:#BDC7D1; --muted:#8B96A2; --rule:#28313A;
  --accent:#63AAE6; --accent-soft:#16303F;
  --negative:#E2827B; --negative-soft:#33211F;
  --positive:#77B896; --positive-soft:#1B2C24;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans","Noto Sans SC","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  font-size:16px; line-height:1.72; -webkit-font-smoothing:antialiased;
}
.page{max-width:1180px; margin:0 auto; padding:0 24px 96px;}
.masthead{
  padding:72px 0 40px; border-bottom:1px solid var(--rule); margin-bottom:44px;
}
.kicker{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
  display:flex; gap:14px; flex-wrap:wrap; align-items:center;
}
.kicker span:not(:first-child)::before{content:"·"; margin-right:14px; color:var(--muted)}
h1{
  font-family:Spectral,Georgia,serif; font-weight:600; font-size:clamp(34px,5.2vw,54px);
  line-height:1.12; margin:18px 0 0; letter-spacing:-.01em; text-wrap:balance; max-width:20ch;
}
.standfirst{
  margin:20px 0 0; max-width:64ch; font-size:19px; line-height:1.66; color:var(--ink-soft);
}
.layout{display:grid; grid-template-columns:1fr; gap:0}
@media (min-width:1000px){ .layout{grid-template-columns:210px minmax(0,1fr); gap:56px} }
nav.index{display:none}
@media (min-width:1000px){
  nav.index{
    display:block; position:sticky; top:28px; align-self:start;
    font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px; line-height:1.55;
  }
  nav.index ol{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:9px}
  nav.index a{color:var(--muted); text-decoration:none; display:grid; grid-template-columns:22px 1fr; gap:6px}
  nav.index a:hover,nav.index a:focus-visible{color:var(--accent)}
  nav.index .n{color:var(--accent); font-variant-numeric:tabular-nums}
}
section{margin:0 0 56px; scroll-margin-top:24px}
h2{
  font-family:Spectral,Georgia,serif; font-weight:600; font-size:26px; line-height:1.28;
  margin:0 0 6px; letter-spacing:-.005em; text-wrap:balance;
}
h2 .n{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:13px; color:var(--accent);
  display:block; letter-spacing:.12em; margin-bottom:8px;
}
h3{font-size:15px; font-weight:600; margin:32px 0 8px; letter-spacing:.01em}
p{margin:0 0 16px; max-width:70ch}
p.lede{color:var(--ink-soft)}
a{color:var(--accent)}
strong{font-weight:600}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.9em}
code{background:var(--surface-sunk); padding:.12em .38em; border-radius:3px}
.formula{
  background:var(--surface); border:1px solid var(--rule); border-left:3px solid var(--accent);
  padding:18px 22px; margin:0 0 20px; font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:14.5px; line-height:1.7; overflow-x:auto; border-radius:0 5px 5px 0;
}
.claims{display:grid; gap:14px; margin:0 0 24px}
@media (min-width:720px){ .claims{grid-template-columns:repeat(3,1fr)} }
.claim{
  background:var(--surface); border:1px solid var(--rule); border-radius:6px;
  padding:18px 20px; box-shadow:var(--shadow);
}
.claim .tag{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; letter-spacing:.14em;
  color:var(--accent); text-transform:uppercase;
}
.claim h4{margin:8px 0 6px; font-size:15px; font-weight:600}
.claim p{margin:0; font-size:14px; color:var(--ink-soft); max-width:none}
.tablewrap{overflow-x:auto; margin:0 0 20px; border:1px solid var(--rule); border-radius:6px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums}
caption{
  caption-side:top; text-align:left; padding:14px 18px 10px; color:var(--muted);
  font-size:12.5px; line-height:1.5;
}
th,td{padding:9px 14px; text-align:right; border-bottom:1px solid var(--rule); white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); font-weight:500; background:var(--surface-sunk);
}
tbody tr:last-child td{border-bottom:none}
td.num,th.num{font-family:"IBM Plex Mono",ui-monospace,monospace}
td.win{color:var(--positive); font-weight:600}
td.bad{color:var(--negative)}
figure{margin:0 0 28px; background:var(--surface); border:1px solid var(--rule); border-radius:6px; padding:16px; box-shadow:var(--shadow)}
figure img{width:100%; height:auto; display:block; border-radius:3px}
figcaption{margin-top:12px; font-size:13px; line-height:1.6; color:var(--muted)}
figcaption b{color:var(--ink-soft); font-weight:600}
.regime{display:flex; align-items:center; gap:10px; margin:3px 0}
.regime .bar{position:relative; height:7px; flex:1; background:var(--surface-sunk); border-radius:4px; overflow:hidden; min-width:90px}
.regime .bar i{position:absolute; inset:0 auto 0 0; display:block; border-radius:4px}
.regime .lbl{font-size:12.5px; color:var(--muted); min-width:132px}
.regime .val{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12.5px; min-width:52px; text-align:right}
.callout{
  border:1px solid var(--rule); border-left:3px solid var(--negative); background:var(--negative-soft);
  padding:16px 20px; border-radius:0 5px 5px 0; margin:0 0 20px;
}
.callout.good{border-left-color:var(--positive); background:var(--positive-soft)}
.callout p{margin:0; max-width:none; font-size:14.5px}
.callout .hd{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--negative); margin-bottom:6px;
}
.callout.good .hd{color:var(--positive)}
ol.steps{padding-left:0; list-style:none; counter-reset:step; display:flex; flex-direction:column; gap:14px; margin:0 0 20px}
ol.steps li{counter-increment:step; display:grid; grid-template-columns:26px 1fr; gap:12px; max-width:70ch}
ol.steps li::before{
  content:counter(step); font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px;
  color:var(--accent); border:1px solid var(--rule); border-radius:4px; height:22px;
  display:grid; place-items:center; margin-top:2px;
}
pre{
  background:var(--surface-sunk); border:1px solid var(--rule); border-radius:5px;
  padding:14px 16px; overflow-x:auto; font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-size:13px; line-height:1.62; margin:0 0 20px;
}
footer{border-top:1px solid var(--rule); padding-top:22px; color:var(--muted); font-size:13px}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style>"""


def regime_bar(label: str, ratio: float, gain: float) -> str:
    """Lambda_rel / Lambda_abs decides everything, so show it directly."""

    width = max(4.0, min(100.0, ratio * 100.0))
    color = "var(--accent)" if ratio < 0.4 else (
        "var(--positive)" if ratio < 0.8 else "var(--negative)"
    )
    return (
        f'<div class="regime"><span class="lbl">{label}</span>'
        f'<span class="bar"><i style="width:{width:.0f}%;background:{color}"></i></span>'
        f'<span class="val">{ratio:.2f}</span>'
        f'<span class="val" style="color:{color}">{gain:.2f}&times;</span></div>'
    )


def main() -> None:
    data = summary()
    regimes = data.get("regimes", [])
    bars = "".join(
        regime_bar(
            f"{r['boundary']} / {r['clutter']}",
            r["occupancy_aligned"] / max(r["occupancy_raw"], 1e-9),
            r["measured_gain"],
        )
        for r in regimes
    )
    html = HEAD + BODY.replace("{{REGIME_BARS}}", bars)
    for slot, name in FIGURE_SLOTS.items():
        html = html.replace(slot, figure(name))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)")


FIGURE_SLOTS = {
    "{{FIG1}}": "fig1_rank_law.png",
    "{{FIG2}}": "fig2_phase_diagram.png",
    "{{FIG3}}": "fig3_carrier_tolerance.png",
    "{{FIG4}}": "fig4_sensor_curves.png",
    "{{FIG5}}": "fig5_task_vs_rank_gain.png",
    "{{FIG6}}": "fig6_fields.png",
    "{{FIG7}}": "fig7_bandwidth_not_frequency.png",
    "{{FIG8}}": "fig8_multicarrier.png",
    "{{FIG9}}": "fig9_estimated_carriers.png",
    "{{FIG10}}": "fig10_learned_baselines.png",
    "{{FIG11}}": "fig11_shifted_pod.png",
    "{{FIG12}}": "fig12_learned_representation.png",
}

from _report_body import BODY

if __name__ == "__main__":
    main()
