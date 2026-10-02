"""Nygen typography and responsive, printable styles for the offline report."""

STYLE = """
:root { color-scheme:light; --blue:#0077fc; --ink:#000000; --muted:#636973;
  --line:#e3e7ec; --paper:#ffffff; --wash:#f5f7fa; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:28px; }
body { margin:0; background:var(--wash); color:var(--ink); font-family:Inter,Arial,sans-serif;
  font-size:15px; font-weight:400; line-height:1.2; letter-spacing:-.04em; }
a { color:var(--blue); text-underline-offset:3px; }
a:focus-visible,summary:focus-visible { outline:3px solid var(--blue); outline-offset:5px; }
.skip-link { position:absolute; top:-100px; left:24px; padding:12px; background:white; }
.skip-link:focus { top:12px; z-index:10; }
.shell { max-width:1280px; margin:0 auto; padding:0 56px; }
.masthead { display:flex; align-items:center; justify-content:space-between; gap:24px;
  padding:28px 0; border-bottom:1px solid var(--line); }
.brand { display:flex; gap:18px; align-items:center; }
.brand img { width:38px; height:38px; object-fit:contain; }
.brand-name { font-size:25px; font-weight:400; }
.brand-name a { color:var(--ink); text-decoration:none; display:inline-block; }
.brand-name a:hover { color:var(--blue); }
.brand-name small { display:block; color:var(--blue); font-size:12px; margin-top:4px; }
.eyebrow { color:#b4b4b4; font-size:12px; font-weight:400; text-transform:uppercase;
  margin:0 0 14px; }
.status { border:1px solid var(--blue); color:var(--blue); border-radius:999px;
  padding:9px 17px; font-size:13px; background:#fff; white-space:nowrap; }
.hero { padding:48px 0 34px; }
h1,h2,h3,p { margin:0; }
h1 { font-size:52px; line-height:1.2; letter-spacing:0; font-weight:400;
  max-width:950px; margin-bottom:16px; }
h2 { font-size:30px; font-weight:400; line-height:1.2; letter-spacing:-.04em; }
h3 { font-size:20px; font-weight:300; line-height:1.2; margin:26px 0 12px; }
.narrative { margin:12px 0; overflow-wrap:anywhere; }
.subtitle { font-size:21px; font-weight:300; max-width:800px; margin:14px 0 18px; }
.study-label { font-size:16px; color:var(--muted); }
.actions { display:flex; flex-wrap:wrap; gap:10px; margin:24px 0 0; }
.button { display:inline-block; border:1px solid var(--blue); border-radius:999px;
  padding:11px 18px; text-decoration:none; color:var(--blue); background:#fff; }
.button.primary { background:var(--blue); color:#fff; }
.contents { display:flex; flex-wrap:wrap; column-gap:25px; row-gap:14px;
  padding:20px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
.contents a { font-size:13px; text-decoration:none; color:var(--muted); }
.contents a:hover { color:var(--blue); }
.contents span { margin-right:6px; color:#969da6; font-size:11px; }
main { padding-bottom:44px; }
section { padding:6px 0; }
.section-number { display:inline-flex; align-items:center; justify-content:center;
  font-size:12px; background:#000; color:#fff; border-radius:999px;
  min-width:37px; height:24px; padding:0 10px; }
p { margin:12px 0; max-width:86ch; }
.muted,.caption { color:var(--muted); }
.caption { font-size:12px; margin-top:12px; }
.notice { padding:20px 23px; border-left:3px solid var(--blue); background:#edf5ff;
  margin:20px 0; }
.notice p { margin:0; }
.notice p + p { margin-top:12px; }
figure { margin:0; padding:18px; background:white; border:1px solid var(--line); }
.embedding { display:block; width:100%; max-height:530px; object-fit:contain; }
.marker-figure { overflow-x:auto; }
.marker-figure:focus-visible { outline:3px solid var(--blue); outline-offset:3px; }
.marker-plot { display:block; width:100%; min-width:640px; height:auto; }
.cluster-size-figure { overflow-x:auto; }
.cluster-size-figure:focus-visible { outline:3px solid var(--blue); outline-offset:3px; }
.cluster-size-plot { display:block; height:298px; width:auto; max-width:none; }
figcaption { margin:12px 0 0; font-size:12px; color:var(--muted); }
.facts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin:0; }
.facts div { padding-bottom:13px; border-bottom:1px solid var(--line); }
dt { color:var(--muted); font-size:12px; margin:0 0 6px; }
dd { margin:0; overflow-wrap:anywhere; }
.table-wrap { width:100%; overflow-x:auto; border:1px solid var(--line); margin:16px 0;
  background:#fff; }
.table-wrap:focus-visible { outline:3px solid var(--blue); outline-offset:3px; }
#populations .step-body > .table-wrap table { min-width:820px; }
#populations .step-body > .table-wrap td:nth-child(2) { min-width:220px; }
.table-hint { display:none; color:var(--muted); font-size:12px; }
table { width:100%; border-collapse:collapse; font-size:13px; text-align:left; }
th { padding:14px 16px; color:var(--muted); font-weight:400; background:#f9fafc;
  border-bottom:1px solid var(--line); vertical-align:bottom; }
td { padding:15px 16px; border-bottom:1px solid var(--line); vertical-align:top;
  overflow-wrap:anywhere; min-width:76px; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover { background:#f7faff; }
td:first-child { font-variant-numeric:tabular-nums; }
details { background:#fff; border:1px solid var(--line); margin:12px 0; }
summary { cursor:pointer; padding:17px 19px; font-weight:400; overflow-wrap:anywhere; }
summary::marker { color:var(--blue); }
details[open] > summary { border-bottom:1px solid var(--line); }
.workflow-step { margin:0; }
.workflow-step > summary { display:grid; grid-template-columns:auto minmax(0,1fr) auto auto;
  align-items:center; column-gap:14px; row-gap:10px; padding:22px; list-style:none; }
.workflow-step > summary::-webkit-details-marker { display:none; }
.workflow-step > summary::after { content:"+"; grid-column:4; grid-row:1;
  color:var(--blue); font-size:24px; width:20px; text-align:center; }
.workflow-step[open] > summary::after { content:"−"; }
.workflow-step > summary h2 { font-size:25px; }
.step-status { font-size:12px; color:var(--muted); }
.step-outcome { grid-column:2 / -1; font-size:14px; color:var(--muted); }
.step-body { padding:22px; }
section:target > .workflow-step { border-color:var(--blue); }
.detail-body { padding:5px 22px 19px; }
.detail-body h3:first-child { margin-top:15px; }
.detail-body .table-wrap { border-left:0; border-right:0; }
ul,ol { padding-left:22px; margin:14px 0; }
li { margin:10px 0; max-width:94ch; overflow-wrap:anywhere; }
.narrative > :first-child { margin-top:0; }
.narrative > :last-child { margin-bottom:0; }
.narrative strong { font-weight:600; }
.narrative code { font-family:ui-monospace,monospace; font-size:.9em;
  letter-spacing:0; background:var(--wash); padding:1px 4px; border-radius:3px; }
.empty { padding:22px; border:1px dashed #cbd2da; color:var(--muted); background:#fff; }
.record-links { display:flex; flex-wrap:wrap; gap:16px; margin:18px 0; }
.record-links a { font-size:13px; }
.footer { display:grid; grid-template-columns:100px minmax(0,1fr); align-items:start;
  gap:24px; padding:25px 0 36px; color:var(--muted); font-size:12px;
  border-top:1px solid var(--line); }
.footer-links { display:flex; flex-wrap:wrap; gap:12px 24px; }
.paper-citation { margin:14px 0; max-width:96ch; }
.footer-note { margin-bottom:0; }
.scarf-logo { background:#000; border-radius:4px; padding:8px 12px; width:100px; height:auto; }
@media (max-width:800px) {
  .table-hint { display:block; }
  .shell { padding:0 24px; } h1 { font-size:38px; letter-spacing:-.04em; }
  .hero { padding-top:34px; }
  .footer { grid-template-columns:1fr; gap:18px; }
  .facts { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .workflow-step > summary { grid-template-columns:auto minmax(0,1fr) auto; padding:18px;
    column-gap:10px; }
  .workflow-step > summary::after { grid-column:3; }
  .workflow-step > summary h2 { font-size:22px; }
  .step-status { grid-column:2; grid-row:2; }
  .step-outcome { grid-column:2 / -1; }
  .step-body { padding:18px; }
  th,td { padding:12px; } h2 { font-size:26px; }
}
@media (max-width:420px) {
  .shell { padding:0 16px; } h1 { font-size:32px; } .brand { gap:10px; }
  .brand-name { font-size:21px; }
  .facts { grid-template-columns:1fr; } .status { padding:8px 12px; }
}
@media print {
  body { background:white; font-size:10pt; } .shell { max-width:none; padding:0; }
  .actions,.contents,.skip-link { display:none; } .hero { padding:20px 0; }
  h1 { font-size:32pt; } h2 { font-size:20pt; }
  section { padding:12px 0; } .table-wrap,.marker-figure { overflow:visible; }
  #populations .step-body > .table-wrap table { min-width:0; }
  #populations .step-body > .table-wrap td:nth-child(2) { min-width:0; }
  .marker-plot { min-width:0; }
  .cluster-size-figure { overflow:visible; }
  .cluster-size-plot { max-width:100%; height:auto; }
  tr,figure,.notice { break-inside:avoid; } thead { display:table-header-group; }
  a { color:#000; } details::details-content { content-visibility:visible; display:block; }
  details > .step-body,details > .detail-body { display:block; }
  .workflow-step > summary::after { display:none; }
}
@media (prefers-reduced-motion:reduce) { html { scroll-behavior:auto; } }
"""
