#!/usr/bin/env python3
import argparse
import copy
import hashlib
import importlib
import json
import re
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
        return {"l": l, "r": r, "t": page_height - t, "b": page_height - b}

    return {"l": l, "r": r, "t": t, "b": b}


def bboxes_are_similar(
    bbox1: dict, bbox2: dict, 
    page_height1: float, page_height2: float, 
    position_tolerance: float,
    size_tolerance: float = None,
) -> bool:
    """
    Check if two bboxes are at similar positions (within tolerance).
    
    Args:
        position_tolerance: Tolerance for left/top position
        size_tolerance: Tolerance for width/height. If None, uses position_tolerance.
    """
    if size_tolerance is None:
        size_tolerance = position_tolerance
        
    norm1 = normalize_bbox_to_topleft(bbox1, page_height1)
    norm2 = normalize_bbox_to_topleft(bbox2, page_height2)
    
    w1 = abs(norm1["r"] - norm1["l"])
    h1 = abs(norm1["b"] - norm1["t"])
    w2 = abs(norm2["r"] - norm2["l"])
    h2 = abs(norm2["b"] - norm2["t"])
    
    return (
        abs(norm1["l"] - norm2["l"]) <= position_tolerance and
        abs(norm1["t"] - norm2["t"]) <= position_tolerance and
        abs(w1 - w2) <= size_tolerance and
        abs(h1 - h2) <= size_tolerance
    )


def is_in_header_region(bbox: dict, page_height: float, header_ratio: float = 0.2) -> bool:
    """
    Check if a bbox is in the header region of a page (top X% of page).
    """
    norm = normalize_bbox_to_topleft(bbox, page_height)
    top = min(norm["t"], norm["b"])  # Get the top edge
    return top < (page_height * header_ratio)


def find_repeated_elements(
    doc_dict: dict,
    min_page_ratio: float = 0.5,
    position_tolerance: float = 5.0,
    size_tolerance: float = 50.0,
) -> Tuple[Set[str], Set[str]]:
    """
    Find elements that appear at similar positions across multiple pages.
    
    Uses pairwise comparison within tolerance.
    
    Args:
        doc_dict: The Docling document dictionary
        min_page_ratio: Minimum ratio of pages an element must appear on to be 
                        considered repeated (0.5 = at least half the pages)
        position_tolerance: Tolerance in points for position comparison
        size_tolerance: Tolerance in points for size comparison (more lenient for header images)
    
    Returns:
        Tuple of:
        - Set of element refs to deduplicate (all except first occurrence)
        - Set of element refs that ARE first occurrences of repeated elements
    """
    pages = doc_dict.get("pages", {})
    page_count = len(pages)
    
    if page_count < 2:
        return set(), set()
    
    min_pages = max(2, int(page_count * min_page_ratio))
    
    # Collect all elements with their position info
    elements: List[dict] = []
    
    def get_page_height(page_no: int) -> float:
        return pages.get(str(page_no), {}).get("size", {}).get("height", 842)
    
    def process_elements(collection: list, element_type: str):
        for idx, elem in enumerate(collection):
            provs = elem.get("prov", [])
            if not provs:
                continue
            
            prov = provs[0]
            page_no = prov.get("page_no")
            bbox = prov.get("bbox")
            
            if not page_no or not bbox:
                continue
            
            page_height = get_page_height(page_no)
            
            elements.append({
                "page_no": page_no,
                "page_height": page_height,
                "bbox": bbox,
                "ref": f"#/{element_type}/{idx}",
                "element_type": element_type,
                "text": elem.get("text", ""),
                "is_header_region": is_in_header_region(bbox, page_height),
            })
    
    process_elements(doc_dict.get("texts", []), "texts")
    process_elements(doc_dict.get("pictures", []), "pictures")
    
    # Group elements by similar position using clustering
    clusters: List[List[dict]] = []
    
    for elem in elements:
        found_cluster = None
        
        # Use more lenient size tolerance for pictures (headers vary)
        elem_size_tol = size_tolerance if elem["element_type"] == "pictures" else position_tolerance
        
        for cluster in clusters:
            rep = cluster[0]
            rep_size_tol = size_tolerance if rep["element_type"] == "pictures" else position_tolerance
            use_size_tol = max(elem_size_tol, rep_size_tol)
            
            if bboxes_are_similar(
                elem["bbox"], rep["bbox"],
                elem["page_height"], rep["page_height"],
                position_tolerance,
                use_size_tol
            ):
                found_cluster = cluster
                break
        
        if found_cluster is not None:
            found_cluster.append(elem)
        else:
            clusters.append([elem])
    
    # Find clusters that span multiple pages
    repeated_refs = set()
    first_occurrence_refs = set()
    
    for cluster in clusters:
        pages_in_cluster = {elem["page_no"] for elem in cluster}
        
        if len(pages_in_cluster) >= min_pages:
            # Sort by page number to identify first occurrence
            sorted_cluster = sorted(cluster, key=lambda x: x["page_no"])
            
            # First occurrence is kept
            first_occurrence_refs.add(sorted_cluster[0]["ref"])
            
            # Rest are duplicates
            for elem in sorted_cluster[1:]:
                repeated_refs.add(elem["ref"])
    
    return repeated_refs, first_occurrence_refs


def get_element_by_ref(doc_dict: dict, ref: str) -> Optional[dict]:
    """Get an element from doc_dict by its ref string."""
    if not ref.startswith("#/"):
        return None
        
    parts = ref[2:].split("/")  # Remove "#/" prefix
    if len(parts) != 2:
        return None
    
    element_type = parts[0]
    try:
        idx = int(parts[1])
    except ValueError:
        return None
    
    collection = doc_dict.get(element_type, [])
    if idx < len(collection):
        return collection[idx]
    return None



def extract_header_content(
    doc_dict: dict, 
    first_occurrence_refs: Set[str],
    repeated_refs: Set[str],
) -> str:
    """
    Extract text content from first-occurrence header elements.
    Only extracts from PAGE 1 to avoid duplicates from variant headers on other pages.

    Returns markdown-formatted header content.
    """
    header_texts: List[str] = []
    seen_texts: Set[str] = set()  # Dedupe by normalized text content

    # Only process first occurrences from page 1
    page1_picture_refs: List[str] = []

    for ref in first_occurrence_refs:
        elem = get_element_by_ref(doc_dict, ref)
        if not elem:
            continue

        provs = elem.get("prov", [])
        if not provs:
            continue

        page_no = provs[0].get("page_no", 0)
        if page_no != 1:
            continue

        if "pictures" in ref:
            page1_picture_refs.append(ref)

    def _format_business_info_line(text: str) -> str:
        """
        Docling sometimes collapses multi-line business-info blocks into one line, e.g.
        'K.v.K.: ... BTW: ... IBAN: ... BIC: ...'.

        Split those into separate lines so the header block stays faithful.
        """
        s = " ".join(text.split())  # normalize whitespace
        # Must contain at least 2 of these labels to be worth splitting
        labels_pat = re.compile(r"(?i)\b(k\.?v\.?k\.?\s*:|btw\s*:|iban\s*:|bic\s*:)")

        matches = list(labels_pat.finditer(s))
        if len(matches) < 2:
            return text.strip()

        def canon(label_raw: str) -> str:
            lr = re.sub(r"\s+", "", label_raw).upper()
            if lr.startswith("K") and "V" in lr and lr.count("K") >= 2:
                return "K.v.K.:"
            if lr.startswith("BTW"):
                return "BTW:"
            if lr.startswith("IBAN"):
                return "IBAN:"
            if lr.startswith("BIC"):
                return "BIC:"
            # Fallback: preserve as-is (trim)
            return label_raw.strip()

        lines: List[str] = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
            label_raw = m.group(1)
            value = s[m.end():end].strip()
            lines.append(f"{canon(label_raw)} {value}".strip())

        return "\n".join(lines).strip()

    def add_text(text: str):
        """Add text if not already seen (by normalized content)."""
        if not text:
            return

        formatted = _format_business_info_line(text.strip())
        normalized = formatted.strip().lower()
        if not normalized or normalized in seen_texts:
            return

        # Check if this text contains or is contained by existing texts
        for existing in list(seen_texts):
            if normalized in existing or existing in normalized:
                if normalized in existing:
                    return
                seen_texts.discard(existing)
                header_texts[:] = [t for t in header_texts if t.strip().lower() != existing]

        seen_texts.add(normalized)
        header_texts.append(formatted.strip())

    # Extract from page 1 pictures' children
    for ref in sorted(page1_picture_refs):
        elem = get_element_by_ref(doc_dict, ref)
        if not elem:
            continue

        children = elem.get("children", [])
        for child_ref_obj in children:
            child_ref = child_ref_obj.get("$ref", "")

            # Skip if this child is marked as deduplicated
            if child_ref in repeated_refs:
                continue

            child_elem = get_element_by_ref(doc_dict, child_ref)
            if child_elem:
                text = (child_elem.get("text") or "").strip()
                if text:
                    add_text(text)

    if not header_texts:
        return ""

    # Sort header texts to put company name first, then address, then business info
    def sort_key(text: str) -> tuple:
        text_lower = text.lower()
        # Company name / logo text (short, distinctive)
        if len(text) < 20 and not any(c.isdigit() for c in text[:5]):
            return (0, text)
        # Address (starts with street name)
        if any(word in text_lower for word in ["allee", "straat", "weg", "laan", "plein"]):
            return (1, text)
        # Postal code / city
        if text[:4].isdigit() or "zwolle" in text_lower:
            return (2, text)
        # Phone
        if text.replace(" ", "").isdigit():
            return (3, text)
        # Email
        if "@" in text:
            return (4, text)
        # Business registration (KvK, BTW, IBAN)
        if any(word in text_lower for word in ["k.v.k", "kvk", "btw", "iban", "bic"]):
            return (5, text)
        return (6, text)

    header_texts.sort(key=sort_key)

    return "\n".join(header_texts)


def deduplicate_document(doc_dict: dict, repeated_refs: Set[str]) -> dict:
    """
    Create a deduplicated copy of the document dictionary.
    
    Elements in repeated_refs are:
    - Moved to content_layer "furniture" 
    - Removed from body.children
    """
    if not repeated_refs:
        return doc_dict
    
    result = copy.deepcopy(doc_dict)
    
    for ref in repeated_refs:
        parts = ref[2:].split("/") if ref.startswith("#/") else ref.split("/")
        if len(parts) != 2:
            continue
        
        element_type = parts[0]
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        
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
    
    return result


def get_repeated_header_texts(doc_dict: dict, repeated_refs: Set[str]) -> Set[str]:
    """
    Get the set of text strings that come from deduplicated header elements.
    These are texts we want to remove from the markdown.
    """
    repeated_texts = set()
    
    for ref in repeated_refs:
        if "texts" not in ref:
            continue
        
        elem = get_element_by_ref(doc_dict, ref)
        if not elem:
            continue
        
        text = elem.get("text", "").strip()
        if text:
            repeated_texts.add(text)
    
    return repeated_texts



def remove_repeated_from_markdown(
    md: str, 
    doc_dict: dict, 
    repeated_refs: Set[str],
    first_occurrence_refs: Set[str],
) -> str:
    """
    Post-process markdown to:
    1. Remove text from repeated header elements (carefully, avoiding false positives)
    2. Remove <!-- image --> markers that correspond to repeated header images
       (deterministically, by mapping markers to picture refs in body order)
    3. Prepend first-occurrence header content at the beginning
    """
    result = md
    image_marker = "<!-- image -->"

    # ----------------------------
    # 1) Remove repeated header texts
    # ----------------------------
    repeated_header_texts = get_repeated_header_texts(doc_dict, repeated_refs)

    # Build set of all body text content (non-header) to avoid false-positive removal
    body_texts: Set[str] = set()
    for idx, elem in enumerate(doc_dict.get("texts", [])):
        ref = f"#/texts/{idx}"

        # Skip elements that are in repeated refs (these are header duplicates)
        if ref in repeated_refs:
            continue
        # Skip elements that are first occurrences of header content
        if ref in first_occurrence_refs:
            continue
        # Skip children of pictures (they're usually header content)
        parent_ref = (elem.get("parent") or {}).get("$ref", "")
        if "pictures" in parent_ref:
            continue

        text = (elem.get("text") or "").strip()
        if text:
            body_texts.add(text)

    # Remove repeated header texts, but ONLY if they appear as standalone lines
    # Remove ALL occurrences (we re-insert the header block from page 1 anyway).
    #
    # Safeguards:
    # - If the exact text exists as body text, do not remove it.
    # - For short strings (e.g. 'Rijff'), also do not remove if it appears as a substring
    #   of any body text (prevents removing from 'A. Rijff').
    for repeated_text in sorted(repeated_header_texts, key=lambda s: (-len(s), s)):
        if repeated_text in body_texts:
            continue

        is_substring_of_body = any(
            repeated_text in body_text and repeated_text != body_text
            for body_text in body_texts
        )
        if len(repeated_text) < 25 and is_substring_of_body:
            continue

        # Remove any line that equals repeated_text (ignoring surrounding whitespace)
        pat = re.compile(
            r"(?m)^[ \t]*" + re.escape(repeated_text.strip()) + r"[ \t]*(?:\r?\n|$)"
        )
        result = pat.sub("", result)

    # ----------------------------
    # 2) Remove image markers for deduplicated header images
    # ----------------------------
    # Determine picture refs in the same order as markdown export (body order is the best proxy).
    picture_flow_refs: List[str] = []
    for ch in (doc_dict.get("body", {}) or {}).get("children", []) or []:
        ref = (ch or {}).get("$ref", "")
        if ref.startswith("#/pictures/"):
            picture_flow_refs.append(ref)

    if not picture_flow_refs:
        picture_flow_refs = [f"#/pictures/{i}" for i in range(len(doc_dict.get("pictures", [])))]

    pic_idx = 0
    new_lines: List[str] = []
    for line in result.split("\n"):
        if line.strip() == image_marker:
            pic_ref = picture_flow_refs[pic_idx] if pic_idx < len(picture_flow_refs) else None
            pic_idx += 1

            # Drop markers for repeated header images
            if pic_ref and pic_ref in repeated_refs:
                continue

            new_lines.append(line)
        else:
            new_lines.append(line)

    result = "\n".join(new_lines)

    # ----------------------------
    # 3) Normalize whitespace
    # ----------------------------
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")

    # ----------------------------
    # 4) Extract and prepend header content
    # ----------------------------
    header_content = extract_header_content(doc_dict, first_occurrence_refs, repeated_refs)

    if header_content:
        # Find where to insert header (after first <!-- image --> if present)
        stripped = result.lstrip()
        if stripped.startswith(image_marker):
            marker_pos = result.find(image_marker)
            marker_end = marker_pos + len(image_marker)
            # Insert header content after the image marker, with proper spacing
            result = (
                result[:marker_end] + 
                "\n\n" + header_content + "\n\n" +
                result[marker_end:]
            )
        else:
            result = header_content + "\n\n" + result

    # Final cleanup
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")

    return result.strip()


def get_deduplication_report(
    doc_dict: dict, 
    repeated_refs: Set[str], 
    first_occurrence_refs: Set[str],
) -> str:
    """Generate a human-readable report of what was deduplicated."""
    lines = []
    
    if first_occurrence_refs:
        lines.append(f"First occurrences (kept as header, {len(first_occurrence_refs)} elements):")
        for ref in sorted(first_occurrence_refs):
            elem = get_element_by_ref(doc_dict, ref)
            if elem:
                text_preview = elem.get("text", "")[:50] or "(picture)"
                prov = (elem.get("prov") or [{}])[0]
                page_no = prov.get("page_no", "?")
                lines.append(f"  + {ref} (page {page_no}): {text_preview!r}...")
    
    if repeated_refs:
        lines.append(f"\nDeduplicated ({len(repeated_refs)} element occurrences removed):")
        for ref in sorted(repeated_refs):
            elem = get_element_by_ref(doc_dict, ref)
            if elem:
                text_preview = elem.get("text", "")[:50] or "(picture)"
                prov = (elem.get("prov") or [{}])[0]
                page_no = prov.get("page_no", "?")
                lines.append(f"  - {ref} (page {page_no}): {text_preview!r}...")
    
    if not lines:
        return "No repeated elements detected."
    
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
    if hasattr(model_cls, "model_fields"):
        return set(model_cls.model_fields.keys())
    if hasattr(model_cls, "__fields__"):
        return set(model_cls.__fields__.keys())
    return None


def validate_dict_keys_against_model(model_instance, data: dict, path: str, strict: bool):
    if not strict:
        return

    model_cls = model_instance.__class__
    fields = _model_fields(model_cls)
    if fields is None:
        return

    unknown = set(data.keys()) - fields
    if unknown:
        raise ValueError(f"Unknown config keys at {path}: {sorted(unknown)}")

    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        child_obj = getattr(model_instance, k, None)
        if child_obj is None:
            continue
        child_fields = _model_fields(child_obj.__class__)
        if child_fields is None:
            continue
        validate_dict_keys_against_model(child_obj, v, f"{path}.{k}", strict)


def make_run_id(profile: str, config: dict) -> str:
    cfg_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    cfg_hash = hashlib.sha1(cfg_bytes).hexdigest()[:8]
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{profile}-{cfg_hash}"


def _allowed_keys(obj) -> Optional[Set[str]]:
    if hasattr(obj.__class__, "model_fields"):
        return set(obj.__class__.model_fields.keys())
    if hasattr(obj.__class__, "__fields__"):
        return set(obj.__class__.__fields__.keys())
    return None


def apply_config_to_object(obj: Any, cfg: Dict[str, Any], path: str, strict: bool):
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} must be an object/dict")

    if strict:
        allowed = _allowed_keys(obj)
        if allowed is not None:
            unknown = set(cfg.keys()) - allowed
            if unknown:
                raise ValueError(f"Unknown config keys at {path}: {sorted(unknown)}")
        else:
            unknown = [k for k in cfg.keys() if not hasattr(obj, k)]
            if unknown:
                raise ValueError(f"Unknown config keys at {path}: {unknown}")

    for k, v in cfg.items():
        cur = getattr(obj, k, None)

        if isinstance(v, dict) and cur is not None:
            apply_config_to_object(cur, v, f"{path}.{k}", strict)
        else:
            setattr(obj, k, v)

    return obj


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("outdir", type=Path, nargs="?", default=Path("out"))

    ap.add_argument("--profile", default="baseline", help="Profile name.")
    ap.add_argument("--configs-dir", type=Path, default=Path("configs"))
    ap.add_argument("--config-file", type=Path, help="Explicit JSON config file.")

    ap.add_argument("--set", action="append", default=[], help="Override config keys.")
    
    # Deduplication options
    ap.add_argument("--no-dedupe", action="store_true", help="Disable deduplication.")
    ap.add_argument("--dedupe-min-ratio", type=float, default=None)
    ap.add_argument("--dedupe-tolerance", type=float, default=None)
    ap.add_argument("--dedupe-size-tolerance", type=float, default=None,
                    help="Size tolerance for picture deduplication (default: 50.0)")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    def load_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    # Load configs
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
        overlay_cfg = load_json(args.config_file)
        loaded_paths.append(str(args.config_file))
    else:
        if profile_path.exists():
            overlay_cfg = load_json(profile_path)
            loaded_paths.append(str(profile_path))
        else:
            if args.profile != "baseline":
                used_fallback = True

    config = deep_merge(base_cfg, overlay_cfg)
    config["profile"] = args.profile

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set must look like key=value, got: {item}")
        k, v = item.split("=", 1)
        set_dotpath(config, k.strip(), parse_value(v.strip()))

    # Get deduplication settings
    dedupe_cfg = config.get("deduplication", {})
    
    dedupe_enabled = not args.no_dedupe
    if "enabled" in dedupe_cfg and not args.no_dedupe:
        dedupe_enabled = dedupe_cfg.get("enabled", True)
    
    dedupe_min_ratio = args.dedupe_min_ratio or dedupe_cfg.get("min_page_ratio", 0.5)
    dedupe_tolerance = args.dedupe_tolerance or dedupe_cfg.get("position_tolerance", 5.0)
    dedupe_size_tolerance = args.dedupe_size_tolerance or dedupe_cfg.get("size_tolerance", 50.0)

    # Run output dir
    run_id = make_run_id(config["profile"], config)
    base = args.pdf.stem
    run_dir = args.outdir / base / config["profile"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build Docling options
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

    pipeline = PdfPipelineOptions()
    apply_config_to_object(pipeline, pipeline_data, "docling.pipeline_options", strict=strict)

    pdf_opt_data = docling_cfg.get("pdf_format_option", {}) or {}
    if not isinstance(pdf_opt_data, dict):
        raise ValueError("config.docling.pdf_format_option must be a JSON object")

    pdf_opt_data = _resolve_special_objects(pdf_opt_data)
    pdf_opt_data = dict(pdf_opt_data)
    pdf_opt_data["pipeline_options"] = pipeline

    default_pdf_opt = PdfFormatOption(pipeline_options=PdfPipelineOptions())
    validate_dict_keys_against_model(
        default_pdf_opt, 
        {k: v for k, v in pdf_opt_data.items() if k != "pipeline_options"},
        "docling.pdf_format_option", 
        strict
    )

    pdf_format_option = PdfFormatOption(**pdf_opt_data)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: pdf_format_option}
    )

    # Convert
    result = converter.convert(str(args.pdf))
    doc = result.document

    # Exports
    doc_dict = doc.export_to_dict()
    md = doc.export_to_markdown()

    # Deduplication
    repeated_refs: Set[str] = set()
    first_occurrence_refs: Set[str] = set()
    deduped_doc_dict = doc_dict
    deduped_md = md
    
    if dedupe_enabled:
        repeated_refs, first_occurrence_refs = find_repeated_elements(
            doc_dict,
            min_page_ratio=dedupe_min_ratio,
            position_tolerance=dedupe_tolerance,
            size_tolerance=dedupe_size_tolerance,
        )
        
        if repeated_refs or first_occurrence_refs:
            print(f"Found {len(first_occurrence_refs)} first-occurrence header elements")
            print(f"Found {len(repeated_refs)} repeated element occurrences to deduplicate")
            
            deduped_doc_dict = deduplicate_document(doc_dict, repeated_refs)
            deduped_md = remove_repeated_from_markdown(
                md, doc_dict, repeated_refs, first_occurrence_refs
            )
            
            report = get_deduplication_report(doc_dict, repeated_refs, first_occurrence_refs)
            (run_dir / f"{base}.dedupe_report.txt").write_text(report, encoding="utf-8")
            print(report)

    # Save outputs
    (run_dir / f"{base}.json").write_text(
        json.dumps(doc_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / f"{base}.md").write_text(md, encoding="utf-8")

    (run_dir / f"{base}.deduped.json").write_text(
        json.dumps(deduped_doc_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / f"{base}.deduped.md").write_text(deduped_md, encoding="utf-8")

    layout_md = recover_linebreaks_in_markdown(args.pdf, deduped_doc_dict, deduped_md)
    (run_dir / f"{base}.layout.md").write_text(layout_md, encoding="utf-8")

    if config.get("exports", {}).get("hover_viewer", False):
        viewer_scale = float(config.get("viewer", {}).get("render_scale", 2.0))
        viewer_dir = export_hover_viewer(run_dir, args.pdf, doc_dict, render_scale=viewer_scale)
        print(f"Viewer written to: {viewer_dir}")

    # Run meta
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
                    "enabled": dedupe_enabled,
                    "min_page_ratio": dedupe_min_ratio,
                    "position_tolerance": dedupe_tolerance,
                    "size_tolerance": dedupe_size_tolerance,
                    "first_occurrence_refs": sorted(first_occurrence_refs),
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