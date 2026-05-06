import cv2
import os
import json
import csv
from datetime import datetime
from modules.detection_logic import PersonTracker
from modules.verification_logic import PostureVerifier

# ── Parameter sets to compare ───────────────────────────────
PARAM_SETS = [
    {
        "name":                "Conservative",
        "confirmation_frames": 8,
        "ratio_threshold":     1.4,
        "angle_min":           35,
        "angle_max":           145,
        "velocity_threshold":  0.03,
    },
    {
        "name":                "Balanced (Default)",
        "confirmation_frames": 5,
        "ratio_threshold":     1.2,
        "angle_min":           45,
        "angle_max":           135,
        "velocity_threshold":  0.02,
    },
    {
        "name":                "Sensitive",
        "confirmation_frames": 3,
        "ratio_threshold":     1.0,
        "angle_min":           55,
        "angle_max":           125,
        "velocity_threshold":  0.01,
    },
    {
        "name":                "Very Sensitive",
        "confirmation_frames": 2,
        "ratio_threshold":     0.9,
        "angle_min":           60,
        "angle_max":           120,
        "velocity_threshold":  0.008,
    },
]


def run_single(video_path, params):
    """Run detection on video with given params, return fall count + frame timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Patch verifier with custom params
    class CustomVerifier(PostureVerifier):
        def __init__(self):
            super().__init__(confirmation_frames=params['confirmation_frames'])
            self.RATIO_THRESHOLD    = params['ratio_threshold']
            self.ANGLE_MIN          = params['angle_min']
            self.ANGLE_MAX          = params['angle_max']
            self.VELOCITY_THRESHOLD = params['velocity_threshold']

    tracker   = PersonTracker(max_persons=3)
    verifiers = {}
    prev_fall_states = {}

    fall_frames  = []
    frame_count  = 0
    fall_count   = 0

    print(f"  Running: {params['name']}...", end='', flush=True)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        any_fall = False
        try:
            if frame_count % 2 == 0:
                persons = tracker.get_persons(frame)
                for p in persons:
                    pid = p['id']
                    if pid not in verifiers:
                        verifiers[pid] = CustomVerifier()
                    is_fall, conf, tif, _ = verifiers[pid].evaluate_posture(p['box'], p['landmarks'])
                    if is_fall:
                        any_fall = True

                was_any = any(prev_fall_states.values()) if prev_fall_states else False
                if any_fall and not was_any:
                    fall_count += 1
                    fall_frames.append(frame_count)

                prev_fall_states = {p['id']: False for p in persons}
                if any_fall:
                    for p in persons:
                        prev_fall_states[p['id']] = True

        except:
            pass

    cap.release()
    print(f" Done. Falls: {fall_count}")

    return {
        "name":         params['name'],
        "params":       params,
        "total_frames": frame_count,
        "fall_count":   fall_count,
        "fall_frames":  fall_frames,
    }


def generate_comparison_report(results, output_dir, video_name):
    """Generate HTML + CSV comparison report."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV ──────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"param_comparison_{ts}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Config', 'Confirmation Frames', 'Ratio Threshold',
            'Angle Min', 'Angle Max', 'Velocity Threshold', 'Falls Detected'
        ])
        for r in results:
            p = r['params']
            writer.writerow([
                r['name'], p['confirmation_frames'], p['ratio_threshold'],
                p['angle_min'], p['angle_max'], p['velocity_threshold'],
                r['fall_count']
            ])

    # ── HTML ─────────────────────────────────────────────────
    max_falls = max(r['fall_count'] for r in results) or 1

    rows_html = ""
    for r in results:
        p = r['params']
        pct = int(r['fall_count'] / max_falls * 100)
        rows_html += f"""
        <tr>
          <td class="name">{r['name']}</td>
          <td>{p['confirmation_frames']}</td>
          <td>{p['ratio_threshold']}</td>
          <td>{p['angle_min']}°–{p['angle_max']}°</td>
          <td>{p['velocity_threshold']}</td>
          <td>
            <div style="display:flex;align-items:center;gap:10px;">
              <div style="flex:1;background:#2d3748;border-radius:4px;height:10px;">
                <div style="width:{pct}%;background:#f97316;height:10px;border-radius:4px;"></div>
              </div>
              <span style="color:#f97316;font-weight:600;min-width:24px">{r['fall_count']}</span>
            </div>
          </td>
        </tr>"""

    # Chart data
    labels    = [r['name'] for r in results]
    falls     = [r['fall_count'] for r in results]
    labels_js = str(labels).replace("'", '"')
    falls_js  = str(falls)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CareBot AI - Parameter Comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; }}
  .header {{ background: #1a1f2e; padding: 2rem 2.5rem; border-bottom: 1px solid #2d3748; }}
  .header h1 {{ font-size: 1.5rem; font-weight: 700; color: #f97316; }}
  .header p  {{ color: #64748b; font-size: 0.85rem; margin-top: 6px; }}
  .container {{ max-width: 1000px; margin: 0 auto; padding: 2rem; }}
  .sec-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 10px; }}
  .card {{ background: #1e2433; border: 1px solid #2d3748; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 10px 14px; color: #64748b; font-size: 11px; text-transform: uppercase; border-bottom: 1px solid #2d3748; }}
  td {{ padding: 12px 14px; border-bottom: 1px solid #1a2030; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #252d3d; }}
  td.name {{ font-weight: 600; color: #f1f5f9; }}
  .insight {{ background: #1a2030; border-left: 3px solid #f97316; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 10px; font-size: 13px; color: #94a3b8; }}
  .insight strong {{ color: #f97316; }}
  canvas {{ max-height: 280px; }}
</style>
</head>
<body>
<div class="header">
  <h1>CareBot AI — Parameter Comparison</h1>
  <p>Video: {video_name} &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp; {results[0]['total_frames']} frames</p>
</div>
<div class="container">

  <p class="sec-label">Falls Detected per Configuration</p>
  <div class="card">
    <canvas id="chart"></canvas>
  </div>

  <p class="sec-label">Parameter Details</p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Configuration</th>
          <th>Confirm Frames</th>
          <th>Ratio Threshold</th>
          <th>Angle Range</th>
          <th>Velocity Threshold</th>
          <th>Falls Detected</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <p class="sec-label">Analysis & Insights</p>
  <div class="card">
    <div class="insight">
      <strong>Conservative:</strong> Fewer false positives, may miss some real falls. Best for environments with lots of normal floor-level activity.
    </div>
    <div class="insight">
      <strong>Balanced (Default):</strong> Optimized trade-off between sensitivity and specificity. Recommended for general use.
    </div>
    <div class="insight">
      <strong>Sensitive:</strong> Catches more falls but may trigger on non-fall events like sitting or bending. Best for high-risk patients.
    </div>
    <div class="insight">
      <strong>Very Sensitive:</strong> Maximum detection rate. Expect higher false positive rate. Use only when missing a fall is unacceptable.
    </div>
  </div>

</div>
<script>
new Chart(document.getElementById('chart'), {{
  type: 'bar',
  data: {{
    labels: {labels_js},
    datasets: [{{
      label: 'Falls Detected',
      data: {falls_js},
      backgroundColor: ['#34d39988','#f9731688','#fbbf2488','#f8717188'],
      borderColor:     ['#34d399',  '#f97316',  '#fbbf24',  '#f87171'],
      borderWidth: 1, borderRadius: 6
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#2d3748' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#2d3748' }}, beginAtZero: true,
            title: {{ display: true, text: 'Falls Detected', color: '#64748b' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    html_path = os.path.join(output_dir, f"param_comparison_{ts}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return html_path, csv_path


def run_comparison(video_path: str, output_dir: str = 'output'):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path)

    print(f"\n[Comparison] Running {len(PARAM_SETS)} configurations on: {video_name}\n")

    results = []
    for params in PARAM_SETS:
        r = run_single(video_path, params)
        if r:
            results.append(r)

    html_path, csv_path = generate_comparison_report(results, output_dir, video_name)

    print(f"\n[Comparison] ✅ Done!")
    print(f"[Comparison] HTML → {html_path}")
    print(f"[Comparison] CSV  → {csv_path}\n")

    return results


if __name__ == "__main__":
    path = input("Enter video path: ").strip('"').strip("'").strip()
    run_comparison(path)