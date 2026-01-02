#!/usr/bin/env python3
import argparse
import copy
import hashlib
import importlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from viewer_export import export_hover_viewer


# ----------------------------
# Repeated element detection
# ----------------------------

def normalize_bbox_to_topleft(bbox: dict, page_height: float) -> dict:
    """
    Normalize bbox to TOPLEFT origin for consistent comparison.
    Returns a new dict with l, t, r, b in TOPLEFT coordinates.
    """
    l, r, t, b = bbox["l"], bbox["r"], bbox["t"], bbox["b"]
    origin = (bbox.get("coord_origin") or "TOPLEFT").upper()

    if origin == "BOTTOMLEFT":
        # Convert: in BOTTOMLEFT, t > b; in TOPLEFT, t < b
        return {"l": l, "r": r, "t": page_height - t, "b": page_height - b}

    return {"l": l, "r": r, "t": t, "b": b}


def bbox_position_key(bbox: dict, page_height: float, tolerance: float = 5.0) -> Tuple[int, int, int, int]:
    """
    Create a quantized position key for grouping similar bboxes.
    Returns (left, top, width, height) rounded to tolerance grid.
    """
    norm = normalize_bbox_to_topleft(bbox, page_height)
    width = abs(norm["r"] - norm["l"])
    height = abs(norm["b"] - norm["t"])
    
    return (
        round(norm["l"] / tolerance),
        round(norm["t"] / tolerance),
        round(width / tolerance),
        round(height / tolerance),
    )


def find_repeated_elements(
    doc_dict: dict,
    min_page_ratio: float = 0.5,
    position_tolerance: float = 5.0,
) -> Set[str]:
    """
    Find elements that appear at similar positions across multiple pages.
    
    Args:
        doc_dict: The Docling document dictionary
        min_page_ratio: Minimum ratio of pages an element must appear on to be 
                        considered repeated (0.5 = at least half the pages)
        position_tolerance: Tolerance in points for position comparison
    
    Returns:
        Set of element refs (e.g., "#/texts/5") that should be deduplicated
        (all occurrences except the first are included)
    """
    pages = doc_dict.get("pages", {})
    page_count = len(pages)
    
    if page_count < 2:
        return set()
    
    min_pages = max(2, int(page_count * min_page_ratio))
    
    # Build mapping: position_key -> list of (page_no, ref, element_type)
    position_groups: Dict[Tuple, List[dict]] = defaultdict(list)
    
    def process_elements(elements: list, element_type: str):
        for idx, elem in enumerate(elements):
            provs = elem.get("prov", [])
            if not provs:
                continue
            
            prov = provs[0]
            page_no = prov.get("page_no")
            bbox = prov.get("bbox")
            
            if not page_no or not bbox:
                continue
            
            page_info = pages.get(str(page_no), {})
            page_height = page_info.get("size", {}).get("height", 842)  # A4 default
            
            pos_key = bbox_position_key(bbox, page_height, position_tolerance)
            
            position_groups[pos_key].append({
                "page_no": page_no,
                "ref": f"#/{element_type}/{idx}",
                "element_type": element_type,
                "text": elem.get("text", "")[:100],  # For debugging
            })
    
    # Process texts and pictures
    process_elements(doc_dict.get("texts", []), "texts")
    process_elements(doc_dict.get("pictures", []), "pictures")
    
    # Find positions that appear on multiple distinct pages
    repeated_refs = set()
    
    for pos_key, occurrences in position_groups.items():
        # Get unique pages where this position appears
        pages_with_element = {occ["page_no"] for occ in occurrences}
        
        if len(pages_with_element) >= min_pages:
            # Sort by page number to keep the first occurrence
            sorted_occs = sorted(occurrences, key=lambda x: x["page_no"])
            
            # Mark all but the first as duplicates
            for occ in sorted_occs[1:]:
                repeated_refs.add(occ["ref"])
    
    return repeated_refs


def deduplicate_document(doc_dict: dict, repeated_refs: Set[str]) -> dict:
    """
    Create a deduplicated copy of the document dictionary.
    
    Elements in repeated_refs are:
    - Moved to content_layer "furniture" 
    - Removed from body.children
    - Added to furniture.children (first occurrence only, rest excluded)
    
    Args:
        doc_dict: Original Docling document dictionary
        repeated_refs: Set of refs to deduplicate (from find_repeated_elements)
    
    Returns:
        New document dictionary with repeated elements handled
    """
    if not repeated_refs:
        return doc_dict
    
    result = copy.deepcopy(doc_dict)
    
    # Track which elements are repeated (including first occurrence)
    # We need to identify ALL occurrences to move the first to furniture
    all_repeated_positions = set()
    
    # Find first occurrences by removing them from repeated_refs logic
    first_occurrence_refs = set()
    
    # Group repeated_refs by their position pattern to find first occurrences
    # This is a bit tricky - we need to re-examine the structure
    
    # Simpler approach: mark repeated elements with content_layer = furniture
    # and filter them from body.children
    
    for ref in repeated_refs:
        parts = ref.split("/")
        if len(parts) != 3:
            continue
        
        element_type = parts[1]  # "texts" or "pictures"
        idx = int(parts[2])
        
        collection = result.get(element_type, [])
        if idx < len(collection):
            collection[idx]["content_layer"] = "furniture"
            collection[idx]["_deduplicated"] = True
    
    # Update body.children to exclude deduplicated elements
    body = result.get("body", {})
    if "children" in body:
        body["children"] = [
            child for child in body["children"]
            if child.get("$ref") not in repeated_refs
        ]
    
    # Optionally: add first occurrences to furniture.children
    # (For now, we just mark them - the markdown export will skip furniture anyway)
    
    return result


def get_deduplication_report(doc_dict: dict, repeated_refs: Set[str]) -> str:
    """
    Generate a human-readable report of what was deduplicated.
    """
    if not repeated_refs:
        return "No repeated elements detected."
    
    lines = [f"Detected {len(repeated_refs)} repeated element occurrences:"]
    
    for ref in sorted(repeated_refs):
        parts = ref.split("/")
        if len(parts) != 3:
            continue
        
        element_type = parts[1]
        idx = int(parts[2])
        
        collection = doc_dict.get(element_type, [])
        if idx < len(collection):
            elem = collection[idx]
            text_preview = elem.get("text", "")[:50]
            prov = (elem.get("prov") or [{}])[0]
            page_no = prov.get("page_no", "?")
            
            lines.append(f"  - {ref} (page {page_no}): {text_preview!r}...")
    
    return "\n".join(lines)


# ----------------------------
# Layout MD recovery helpers
# ----------------------------

def bbox_to_fitz_rect(bbox: dict, page: fitz.Page) -> fitz.Rect:
    l, r, t, b = bbox["l"], bbox["r"], bbox["t"], bbox["b"]
    origin = (bbox.get("coord_origin") or "TOPLEFT").upper()

    if origin == "BOTTOMLEFT":
        h = page.rect.height
        return fitz.Rect(l, h - t, r, h - b)

    return fitz.Rect(l, t, r, b)


def clean_raw_text(raw: str) -> str:
    raw = raw.replace("\u200b", "").replace("\ufeff", "")
    lines = [ln.rstrip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln.strip() != ""]
    return "\n".join(lines)


def recover_linebreaks_in_markdown(pdf_path: Path, doc_dict: dict, md: str) -> str:
    pdf = fitz.open(str(pdf_path))
    out = md

    for t in doc_dict.get("texts", []):
        text = t.get("text") or ""
        provs = t.get("prov") or []
        if not provs:
            continue

        prov0 = provs[0]
        page_idx = prov0["page_no"] - 1
        page = pdf[page_idx]

        rect = bbox_to_fitz_rect(prov0["bbox"], page)
        raw = page.get_text("text", clip=rect)
        raw = clean_raw_text(raw)

        if "\n" in raw and "\n" not in text and text:
            raw_md = "  \n".join(raw.splitlines())
            if out.count(text) == 1:
                out = out.replace(text, raw_md, 1)

    pdf.close()
    return out


# ----------------------------
# Markdown export with deduplication
# ----------------------------

def export_markdown_deduplicated(doc_dict: dict, repeated_refs: Set[str]) -> str:
    """
    Export markdown while excluding repeated elements.
    
    This is a simplified approach - we filter the texts that would be 
    exported based on the repeated_refs set.
    """
    # For a proper implementation, we'd need to walk the document tree
    # For now, we'll use Docling's built-in export on a filtered document
    
    # The deduplicate_document function already marks elements,
    # but Docling's export_to_markdown doesn't respect our custom flags.
    # 
    # Alternative: manually build markdown from the document structure,
    # or post-process the markdown to remove duplicates.
    
    # For now, return None to indicate we should use post-processing
    return None


def remove_repeated_from_markdown(md: str, doc_dict: dict, repeated_refs: Set[str]) -> str:
    """
    Post-process markdown to remove text from repeated elements.
    
    This is a best-effort approach that removes exact text matches.
    """
    result = md
    
    for ref in repeated_refs:
        parts = ref.split("/")
        if len(parts) != 3:
            continue
        
        element_type = parts[1]
        idx = int(parts[2])
        
        collection = doc_dict.get(element_type, [])
        if idx >= len(collection):
            continue
        
        elem = collection[idx]
        text = elem.get("text", "").strip()
        
        if not text:
            continue
        
        # Try to remove exact matches (be careful with partial matches)
        # Only remove if it appears as a whole line or paragraph
        for pattern in [
            f"\n{text}\n",      # Standalone paragraph
            f"\n{text}",        # End of document
            f"{text}\n",        # Start of document
        ]:
            if pattern in result:
                result = result.replace(pattern, "\n", 1)
                break
    
    # Clean up multiple blank lines
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    
    return result.strip()


# ----------------------------
# Config / model utilities
# ----------------------------

def deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge b onto a; scalars/lists overwrite."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        pass
    if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
        return json.loads(v)
    return v


def set_dotpath(d: dict, path: str, value):
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _import_symbol(dotted_path: str):
    mod, _, name = dotted_path.rpartition(".")
    if not mod:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    m = importlib.import_module(mod)
    return getattr(m, name)


def _resolve_special_objects(obj: Any) -> Any:
    """
    Allows config literals like:
      {"__class__": "docling.datamodel.pipeline_options.TesseractCliOcrOptions", ...}
      {"__enum__": "docling.datamodel.pipeline_options.TableFormerMode.ACCURATE"}
    """
    if isinstance(obj, list):
        return [_resolve_special_objects(x) for x in obj]

    if isinstance(obj, dict):
        if "__type__" in obj:
            return _import_symbol(obj["__type__"])
        
        if "__enum__" in obj:
            enum_path = obj["__enum__"]
            enum_cls_path, _, member = enum_path.rpartition(".")
            enum_cls = _import_symbol(enum_cls_path)
            return getattr(enum_cls, member)

        if "__class__" in obj:
            cls_path = obj["__class__"]
            cls = _import_symbol(cls_path)
            kwargs = {k: _resolve_special_objects(v) for k, v in obj.items() if k != "__class__"}

            if hasattr(cls, "model_validate"):
                return cls.model_validate(kwargs)
            if hasattr(cls, "parse_obj"):
                return cls.parse_obj(kwargs)
            return cls(**kwargs)

        return {k: _resolve_special_objects(v) for k, v in obj.items()}

    return obj


def _model_fields(model_cls) -> Optional[Set[str]]:
    # pydantic v2
    if hasattr(model_cls, "model_fields"):
        return set(model_cls.model_fields.keys())
    # pydantic v1
    if hasattr(model_cls, "__fields__"):
        return set(model_cls.__fields__.keys())
    return None


def validate_dict_keys_against_model(model_instance, data: dict, path: str, strict: bool):
    """
    Best-effort strict key validation:
    - checks unknown keys at current level
    - recurses into nested option objects (accelerator_options, layout_options, etc.)
      by looking at the *type* of the default attribute on the model instance.
    """
    if not strict:
        return

    model_cls = model_instance.__class__
    fields = _model_fields(model_cls)
    if fields is None:
        return  # can't validate

    unknown = set(data.keys()) - fields
    if unknown:
        raise ValueError(f"Unknown config keys at {path}: {sorted(unknown)}")

    # recurse into nested pydantic-ish objects
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        child_obj = getattr(model_instance, k, None)
        if child_obj is None:
            continue
        # nested model?
        child_fields = _model_fields(child_obj.__class__)
        if child_fields is None:
            continue
        validate_dict_keys_against_model(child_obj, v, f"{path}.{k}", strict)


def build_pydantic_model(model_cls, data: dict, path: str, strict: bool):
    """
    Validate keys (strict) then model_validate/parse_obj.
    """
    # Create a default instance for recursion-based key validation
    try:
        default_inst = model_cls()  # works for PdfPipelineOptions and nested options
    except TypeError:
        default_inst = None

    if default_inst is not None:
        validate_dict_keys_against_model(default_inst, data, path, strict)

    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    if hasattr(model_cls, "parse_obj"):
        return model_cls.parse_obj(data)
    return model_cls(**data)


def make_run_id(profile: str, config: dict) -> str:
    cfg_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    cfg_hash = hashlib.sha1(cfg_bytes).hexdigest()[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{profile}-{cfg_hash}"

def _allowed_keys(obj) -> Optional[Set[str]]:
    # pydantic v2
    if hasattr(obj.__class__, "model_fields"):
        return set(obj.__class__.model_fields.keys())
    # pydantic v1
    if hasattr(obj.__class__, "__fields__"):
        return set(obj.__class__.__fields__.keys())
    return None


def apply_config_to_object(obj: Any, cfg: Dict[str, Any], path: str, strict: bool):
    """
    Recursively apply cfg onto obj WITHOUT replacing nested option objects by default.
    If a value is a dict, we recurse into the existing attribute object.
    If a value is a concrete object (e.g. produced by __class__/__type__), we replace the attribute.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must be an object/dict")

    if strict:
        allowed = _allowed_keys(obj)
        if allowed is not None:
            unknown = set(cfg.keys()) - allowed
            if unknown:
                raise ValueError(f"Unknown config keys at {path}: {sorted(unknown)}")
        else:
            # non-pydantic object: best-effort attribute check
            unknown = [k for k in cfg.keys() if not hasattr(obj, k)]
            if unknown:
                raise ValueError(f"Unknown config keys at {path}: {unknown}")

    for k, v in cfg.items():
        cur = getattr(obj, k, None)

        if isinstance(v, dict) and cur is not None:
            # Deep-apply into existing nested options object
            apply_config_to_object(cur, v, f"{path}.{k}", strict)
        else:
            # Scalars, lists, or already-resolved objects (from __class__/__type__/__enum__)
            setattr(obj, k, v)

    return obj



# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("outdir", type=Path, nargs="?", default=Path("out"))

    ap.add_argument("--profile", default="baseline", help="Profile name (loads configs/<profile>.json).")
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
    ap.add_argument("--config-file", type=Path, help="Explicit JSON file to overlay on baseline (optional).")

    ap.add_argument("--set", action="append", default=[], help="Override config keys via dotpath, e.g. --set docling.pipeline_options.do_ocr=false")
    
    # Deduplication options
    ap.add_argument("--no-dedupe", action="store_true", help="Disable repeated header/footer deduplication.")
    ap.add_argument("--dedupe-min-ratio", type=float, default=0.5, 
                    help="Minimum page ratio for element to be considered repeated (default: 0.5)")
    ap.add_argument("--dedupe-tolerance", type=float, default=5.0,
                    help="Position tolerance in points for deduplication (default: 5.0)")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    def load_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    # ---- Load baseline + overlay (deep-merge) ----
    baseline_path = args.configs_dir / "baseline.json"
    profile_path = args.configs_dir / f"{args.profile}.json"

    loaded_paths = []
    used_fallback = False

    base_cfg = {}
    if baseline_path.exists():
        base_cfg = load_json(baseline_path)
        loaded_paths.append(str(baseline_path))

    overlay_cfg = {}
    if args.config_file:
        # Explicit overlay file on top of baseline
        overlay_cfg = load_json(args.config_file)
        loaded_paths.append(str(args.config_file))
    else:
        # Profile overlay, else fallback to baseline only
        if profile_path.exists():
            overlay_cfg = load_json(profile_path)
            loaded_paths.append(str(profile_path))
        else:
            if args.profile != "baseline":
                used_fallback = True  # asked for tuned profile, but only baseline exists
            overlay_cfg = {}

    config = deep_merge(base_cfg, overlay_cfg)

    # Force profile id to the CLI value (folder grouping key)
    config["profile"] = args.profile

    # Apply --set overrides last
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set must look like key=value, got: {item}")
        k, v = item.split("=", 1)
        set_dotpath(config, k.strip(), parse_value(v.strip()))

    # ---- Run output dir ----
    run_id = make_run_id(config["profile"], config)
    base = args.pdf.stem
    run_dir = args.outdir / base / config["profile"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build Docling options from config (pass-through) ----
    # All Docling settings should live under: config["docling"]
    docling_cfg = copy.deepcopy(config.get("docling", {}))
    docling_cfg = _resolve_special_objects(docling_cfg)
    strict = bool(docling_cfg.get("strict", True))

    pipeline_data = docling_cfg.get("pipeline_options", {})
    if pipeline_data.get("do_ocr", False):
        oo = pipeline_data.get("ocr_options")
    if isinstance(oo, dict) and "kind" not in oo and "__class__" not in oo:
        oo["kind"] = docling_cfg.get("ocr_engine", "auto")
        pipeline_data["ocr_options"] = oo
    if not isinstance(pipeline_data, dict):
        raise ValueError("config.docling.pipeline_options must be a JSON object")

    # pipeline = build_pydantic_model(PdfPipelineOptions, pipeline_data, "docling.pipeline_options", strict=strict)
    pipeline = PdfPipelineOptions()
    apply_config_to_object(pipeline, pipeline_data, "docling.pipeline_options", strict=strict)

    # PdfFormatOption: allow setting backend and other pdf-format-specific knobs
    pdf_opt_data = docling_cfg.get("pdf_format_option", {})
    if pdf_opt_data is None:
        pdf_opt_data = {}
    if not isinstance(pdf_opt_data, dict):
        raise ValueError("config.docling.pdf_format_option must be a JSON object")

    pdf_opt_data = _resolve_special_objects(pdf_opt_data)
    pdf_opt_data = dict(pdf_opt_data)
    pdf_opt_data["pipeline_options"] = pipeline

    # Validate keys for PdfFormatOption (requires a pipeline_options instance)
    default_pdf_opt = PdfFormatOption(pipeline_options=PdfPipelineOptions())
    validate_dict_keys_against_model(default_pdf_opt, {k: v for k, v in pdf_opt_data.items() if k != "pipeline_options"},
                                     "docling.pdf_format_option", strict)

    pdf_format_option = PdfFormatOption(**pdf_opt_data)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: pdf_format_option}
    )

    # ---- Convert ----
    result = converter.convert(str(args.pdf))
    doc = result.document

    # ---- Exports ----
    doc_dict = doc.export_to_dict()
    md = doc.export_to_markdown()

    # ---- Deduplication ----
    repeated_refs: Set[str] = set()
    deduped_doc_dict = doc_dict
    deduped_md = md
    
    if not args.no_dedupe:
        repeated_refs = find_repeated_elements(
            doc_dict,
            min_page_ratio=args.dedupe_min_ratio,
            position_tolerance=args.dedupe_tolerance,
        )
        
        if repeated_refs:
            print(f"Found {len(repeated_refs)} repeated element occurrences to deduplicate")
            
            # Create deduplicated document dict
            deduped_doc_dict = deduplicate_document(doc_dict, repeated_refs)
            
            # Create deduplicated markdown
            deduped_md = remove_repeated_from_markdown(md, doc_dict, repeated_refs)
            
            # Write deduplication report
            report = get_deduplication_report(doc_dict, repeated_refs)
            (run_dir / f"{base}.dedupe_report.txt").write_text(report, encoding="utf-8")
            print(report)

    # Save original (non-deduplicated) outputs
    (run_dir / f"{base}.json").write_text(json.dumps(doc_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / f"{base}.md").write_text(md, encoding="utf-8")

    # Save deduplicated outputs
    (run_dir / f"{base}.deduped.json").write_text(json.dumps(deduped_doc_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / f"{base}.deduped.md").write_text(deduped_md, encoding="utf-8")

    # Layout recovery on deduplicated markdown
    layout_md = recover_linebreaks_in_markdown(args.pdf, deduped_doc_dict, deduped_md)
    (run_dir / f"{base}.layout.md").write_text(layout_md, encoding="utf-8")

    if config.get("exports", {}).get("hover_viewer", False):
        viewer_scale = float(config.get("viewer", {}).get("render_scale", 2.0))
        viewer_dir = export_hover_viewer(run_dir, args.pdf, doc_dict, render_scale=viewer_scale)
        print(f"Viewer written to: {viewer_dir}")

    # ---- Run meta ----
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "pdf": str(args.pdf),
                "profile": config["profile"],
                "run_id": run_id,
                "configs_dir": str(args.configs_dir),
                "loaded_config_files": loaded_paths,
                "used_fallback": used_fallback,
                "config": config,
                "docling_status": str(result.status),
                "deduplication": {
                    "enabled": not args.no_dedupe,
                    "min_page_ratio": args.dedupe_min_ratio,
                    "position_tolerance": args.dedupe_tolerance,
                    "repeated_elements_found": len(repeated_refs),
                    "repeated_refs": sorted(repeated_refs),
                },
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote run outputs to: {run_dir}")
    if loaded_paths:
        print("Loaded config files:")
        for p in loaded_paths:
            print(f"  - {p}")
    if used_fallback:
        print("NOTE: profile config was missing; ran with baseline only.")


if __name__ == "__main__":
    main()