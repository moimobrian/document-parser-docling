# viewer_export.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF


def export_hover_viewer(
    run_dir: Path,
    pdf_path: Path,
    doc_dict: Dict[str, Any],
    render_scale: float = 2.0,
) -> Path:
    """
    Creates:
      run_dir/viewer/index.html
      run_dir/viewer/viewer_data.json
      run_dir/viewer/pages/page-001.png ...
    Returns the viewer directory Path.
    """
    viewer_dir = run_dir / "viewer"
    pages_dir = viewer_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(str(pdf_path))
    pages_meta = []
    items = []

    def bbox_to_px_rect(bbox: dict, page: fitz.Page, scale: float) -> dict:
        l, r, t, b = bbox["l"], bbox["r"], bbox["t"], bbox["b"]
        origin = (bbox.get("coord_origin") or "TOPLEFT").upper()
        page_h = page.rect.height

        if origin == "BOTTOMLEFT":
            y0 = (page_h - t) * scale
            y1 = (page_h - b) * scale
        else:
            y0 = t * scale
            y1 = b * scale

        x0 = l * scale
        x1 = r * scale

        return {
            "x": float(min(x0, x1)),
            "y": float(min(y0, y1)),
            "w": float(abs(x1 - x0)),
            "h": float(abs(y1 - y0)),
        }

    # Render pages
    for i in range(pdf.page_count):
        page = pdf.load_page(i)
        mat = fitz.Matrix(render_scale, render_scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_name = f"page-{i+1:03d}.png"
        img_path = pages_dir / img_name
        pix.save(str(img_path))

        pages_meta.append(
            {"page_no": i + 1, "img": f"pages/{img_name}", "img_w": pix.width, "img_h": pix.height}
        )

    # Text boxes
    for t in doc_dict.get("texts", []):
        text = (t.get("text") or "").strip()
        provs = t.get("prov") or []
        if not provs:
            continue
        p = provs[0]
        page_no = p["page_no"]
        page = pdf.load_page(page_no - 1)
        rect = bbox_to_px_rect(p["bbox"], page, render_scale)

        area = rect["w"] * rect["h"]
        items.append({
            "id": f"text/{len(items)}",
            "kind": "text",
            "page_no": page_no,
            "label": "text",
            "text": text[:300],
            "area": area,
            **rect,
        })

    # Picture boxes
    for pid, pic in enumerate(doc_dict.get("pictures", [])):
        provs = pic.get("prov") or []
        if not provs:
            continue
        p = provs[0]
        page_no = p["page_no"]
        page = pdf.load_page(page_no - 1)
        rect = bbox_to_px_rect(p["bbox"], page, render_scale)

        area = rect["w"] * rect["h"]
        items.append({
            "id": f"picture/{pid}",
            "kind": "picture",
            "page_no": page_no,
            "label": f"picture/{pid}",
            "text": "",
            "area": area,
            **rect,
        })

    pdf.close()

    data = {"pages": pages_meta, "items": items}
    (viewer_dir / "viewer_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Docling Hover Viewer</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; margin: 16px; }
    img { display: block; }
    .top { display: flex; gap: 12px; align-items: center; margin-bottom: 12px; }
    .canvas { position: relative; display: inline-block; border: 1px solid #ddd; }
    .box { position: absolute; border: 2px solid rgba(0,120,255,0.7); box-sizing: border-box; }
    .box.picture { pointer-events: none; border-style: dashed; opacity: 0.7; }
    .tooltip {
      position: fixed; pointer-events: none;
      background: rgba(0,0,0,0.85); color: #fff;
      padding: 6px 8px; border-radius: 6px; font-size: 12px;
      max-width: 520px; white-space: pre-wrap; display: none;
    }
  </style>
</head>
<body>
  <div class="top">
    <label>Page:
      <select id="pageSel"></select>
    </label>
    <span id="counts"></span>
  </div>
  <div class="canvas" id="canvas">
    <img id="pageImg" />
  </div>
  <div class="tooltip" id="tip"></div>

<script>
(async function(){
  const data = await fetch("./viewer_data.json").then(r => r.json());
  const pageSel = document.getElementById("pageSel");
  const canvas = document.getElementById("canvas");
  const img = document.getElementById("pageImg");
  const tip = document.getElementById("tip");
  const counts = document.getElementById("counts");

  data.pages.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.page_no;
    opt.textContent = p.page_no;
    pageSel.appendChild(opt);
  });

  function showTip(ev, item){
    tip.style.display = "block";
    tip.style.left = (ev.clientX + 12) + "px";
    tip.style.top = (ev.clientY + 12) + "px";
    tip.textContent = item.kind + " | " + item.label + (item.text ? ("\\n\\n" + item.text) : "");
  }
  function hideTip(){ tip.style.display = "none"; }

  function render(pageNo){
    Array.from(canvas.querySelectorAll(".box")).forEach(x => x.remove());

    const page = data.pages.find(p => p.page_no == pageNo);
    img.src = page.img;
    img.width = page.img_w;
    img.height = page.img_h;

    img.onload = () => {
        const sx = img.clientWidth / img.naturalWidth;
        const sy = img.clientHeight / img.naturalHeight;

        const pageItems = data.items.filter(it => it.page_no == pageNo);
        pageItems.sort((a, b) => (b.area || 0) - (a.area || 0));

        counts.textContent = `${pageItems.length} items`;

        pageItems.forEach(it => {
            const d = document.createElement("div");
            d.className = "box";
            d.className = "box " + it.kind;
            d.style.left = it.x + "px";
            d.style.top = it.y + "px";
            d.style.width = it.w + "px";
            d.style.height = it.h + "px";
            d.addEventListener("mousemove", (ev) => showTip(ev, it));
            d.addEventListener("mouseleave", hideTip);
            canvas.appendChild(d);
        });
    };
  };
  

  pageSel.addEventListener("change", () => render(parseInt(pageSel.value, 10)));
  render(parseInt(pageSel.value || "1", 10));
})();
</script>
</body>
</html>"""
    (viewer_dir / "index.html").write_text(html, encoding="utf-8")

    return viewer_dir
