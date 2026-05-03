import os
import base64
from datetime import datetime


def _img_to_base64(img_path):
    """Convert image file to base64 string for embedding in HTML."""
    try:
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None


def generate_report(session_data: dict, output_dir='output'):
    """
    Generate a professional HTML report after each session.

    session_data keys:
        - start_time     : datetime
        - end_time       : datetime
        - total_frames   : int
        - fps            : float
        - falls          : list of dicts:
                           { 'id': int, 'person_id': int, 'timestamp': str,
                             'frame': int, 'screenshot': str (path) }
        - fall_timeline  : list of 0/1 per frame
        - video_source   : str
    """
    os.makedirs(output_dir, exist_ok=True)

    start        = session_data.get('start_time', datetime.now())
    end          = session_data.get('end_time',   datetime.now())
    duration     = (end - start).total_seconds()
    total_frames = session_data.get('total_frames', 0)
    fps          = session_data.get('fps', 0)
    falls        = session_data.get('falls', [])
    timeline     = session_data.get('fall_timeline', [])
    source       = session_data.get('video_source', 'Unknown')

    mins, secs   = divmod(int(duration), 60)
    fall_rate    = (len(falls) / total_frames * 100) if total_frames > 0 else 0

    # Build timeline SVG
    tl_svg = _build_timeline_svg(timeline)

    # Build fall cards HTML
    fall_cards_html = _build_fall_cards(falls)

    report_time = end.strftime("%Y-%m-%d %H:%M:%S")
    file_ts     = end.strftime("%Y%m%d_%H%M%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CareBot AI - Session Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%); padding: 2.5rem 3rem; border-bottom: 1px solid #2d3748; }}
  .header h1 {{ font-size: 2rem; font-weight: 700; color: #f97316; letter-spacing: -0.5px; }}
  .header p  {{ color: #94a3b8; margin-top: 6px; font-size: 0.9rem; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-left: 12px; vertical-align: middle; }}
  .badge-ok   {{ background: #064e3b; color: #6ee7b7; }}
  .badge-warn {{ background: #7f1d1d; color: #fca5a5; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2.5rem 2rem; }}
  .section-title {{ font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 1rem; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
  .stat-card {{ background: #1e2433; border: 1px solid #2d3748; border-radius: 12px; padding: 1.25rem 1.5rem; }}
  .stat-card .label {{ font-size: 0.75rem; color: #64748b; margin-bottom: 6px; }}
  .stat-card .value {{ font-size: 1.75rem; font-weight: 700; color: #f1f5f9; }}
  .stat-card .value.danger {{ color: #f87171; }}
  .stat-card .value.safe   {{ color: #34d399; }}
  .stat-card .value.accent {{ color: #f97316; }}
  .card {{ background: #1e2433; border: 1px solid #2d3748; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
  .timeline-wrap {{ overflow-x: auto; }}
  .falls-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
  .fall-card {{ background: #1a1220; border: 1px solid #4c1d3a; border-radius: 10px; overflow: hidden; }}
  .fall-card .fc-header {{ background: #2d1022; padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; }}
  .fall-card .fc-title  {{ font-size: 0.85rem; font-weight: 600; color: #f87171; }}
  .fall-card .fc-time   {{ font-size: 0.75rem; color: #94a3b8; }}
  .fall-card img {{ width: 100%; display: block; max-height: 180px; object-fit: cover; }}
  .fall-card .fc-body {{ padding: 10px 14px; font-size: 0.8rem; color: #94a3b8; }}
  .no-falls {{ text-align: center; padding: 3rem; color: #34d399; font-size: 1.1rem; }}
  .footer {{ text-align: center; padding: 2rem; color: #334155; font-size: 0.8rem; border-top: 1px solid #1e2433; margin-top: 2rem; }}
  .source-path {{ font-family: monospace; font-size: 0.8rem; color: #475569; background: #0f1117; padding: 4px 10px; border-radius: 6px; }}
</style>
</head>
<body>

<div class="header">
  <h1>CareBot AI
    <span class="badge {'badge-warn' if falls else 'badge-ok'}">
      {'⚠ ' + str(len(falls)) + ' Fall(s) Detected' if falls else '✓ No Falls Detected'}
    </span>
  </h1>
  <p>Session Report &nbsp;·&nbsp; {report_time} &nbsp;·&nbsp; <span class="source-path">{os.path.basename(str(source))}</span></p>
</div>

<div class="container">

  <!-- STATS -->
  <p class="section-title">Session Summary</p>
  <div class="stats-grid">
    <div class="stat-card">
      <div class="label">Total Falls</div>
      <div class="value {'danger' if falls else 'safe'}">{len(falls)}</div>
    </div>
    <div class="stat-card">
      <div class="label">Duration</div>
      <div class="value accent">{mins:02d}:{secs:02d}</div>
    </div>
    <div class="stat-card">
      <div class="label">Frames Processed</div>
      <div class="value">{total_frames:,}</div>
    </div>
    <div class="stat-card">
      <div class="label">Avg FPS</div>
      <div class="value">{fps:.1f}</div>
    </div>
    <div class="stat-card">
      <div class="label">Fall Rate</div>
      <div class="value {'danger' if fall_rate > 5 else 'safe'}">{fall_rate:.2f}%</div>
    </div>
    <div class="stat-card">
      <div class="label">Session Start</div>
      <div class="value" style="font-size:1rem">{start.strftime('%H:%M:%S')}</div>
    </div>
  </div>

  <!-- TIMELINE -->
  <p class="section-title">Fall Timeline</p>
  <div class="card">
    <div class="timeline-wrap">
      {tl_svg}
    </div>
    <p style="font-size:0.75rem; color:#475569; margin-top:10px;">
      Each bar = 1 frame &nbsp;·&nbsp; 
      <span style="color:#f87171">■</span> Fall &nbsp; 
      <span style="color:#1e4034">■</span> Normal
    </p>
  </div>

  <!-- FALL EVENTS -->
  <p class="section-title">Fall Events ({len(falls)})</p>
  <div class="card">
    {'<div class="falls-grid">' + fall_cards_html + '</div>' if falls else '<div class="no-falls">✓ No falls detected in this session</div>'}
  </div>

</div>

<div class="footer">Generated by CareBot AI &nbsp;·&nbsp; {report_time}</div>
</body>
</html>"""

    report_path = os.path.join(output_dir, f'report_{file_ts}.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[Report] ✅ Report saved: {report_path}")
    return report_path


def _build_timeline_svg(timeline):
    if not timeline:
        return '<p style="color:#475569">No timeline data.</p>'

    w, h = 1000, 60
    bar_w = max(1, w // max(len(timeline), 1))

    bars = ''
    for i, val in enumerate(timeline):
        color = '#f87171' if val == 1 else '#1e4034'
        bh    = h if val == 1 else 8
        by    = h - bh
        bars += f'<rect x="{i*bar_w}" y="{by}" width="{max(1,bar_w-1)}" height="{bh}" fill="{color}"/>'

    return f'''<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"
               style="width:100%;height:60px;display:block;">
      {bars}
    </svg>'''


def _build_fall_cards(falls):
    if not falls:
        return ''

    cards = ''
    for f in falls:
        person_id  = f.get('person_id', 0) + 1
        timestamp  = f.get('timestamp', 'N/A')
        frame_num  = f.get('frame', 'N/A')
        screenshot = f.get('screenshot', None)

        img_html = ''
        if screenshot and os.path.exists(screenshot):
            b64 = _img_to_base64(screenshot)
            if b64:
                img_html = f'<img src="data:image/jpeg;base64,{b64}" alt="Fall evidence"/>'

        cards += f'''
        <div class="fall-card">
          <div class="fc-header">
            <span class="fc-title">⚠ Person #{person_id} - Fall Detected</span>
            <span class="fc-time">{timestamp}</span>
          </div>
          {img_html}
          <div class="fc-body">Frame #{frame_num}</div>
        </div>'''

    return cards