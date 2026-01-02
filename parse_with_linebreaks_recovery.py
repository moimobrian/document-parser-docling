#!/usr/bin/env python3
import argparse
import copy
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import fitz  # PyMuPDF

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from viewer_export import export_hover_viewer


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

    (run_dir / f"{base}.json").write_text(json.dumps(doc_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / f"{base}.md").write_text(md, encoding="utf-8")

    layout_md = recover_linebreaks_in_markdown(args.pdf, doc_dict, md)
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
