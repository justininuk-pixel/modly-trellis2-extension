import io
import os
import sys
import time
import threading
import uuid
import traceback
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_EXTENSION_DIR = Path(__file__).parent
_VENDOR_DIR    = _EXTENSION_DIR / "vendor"


class Trellis2Generator(BaseGenerator):
    MODEL_ID     = "trellis-2"
    DISPLAY_NAME = "TRELLIS.2"
    VRAM_GB      = 24

    def is_downloaded(self) -> bool:
        return (self.model_dir / "pipeline.json").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._auto_download()

        self._setup_env()
        self._setup_vendor()

        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except ImportError as e:
            raise RuntimeError(f"[TRELLIS] Import failed: {e}")

        print("[TRELLIS] Loading model...")

        try:
            pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(self.model_dir))
            pipe.cuda()
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"[TRELLIS] Model load failed: {e}")

        self._model = pipe
        print("[TRELLIS] Model loaded.")

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        try:
            return self._generate_impl(image_bytes, params, progress_cb, cancel_event)
        except GenerationCancelled:
            raise
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"TRELLIS failed: {e}")

    def _generate_impl(self, image_bytes, params, progress_cb, cancel_event):

        if self._model is None:
            self.load()

        try:
            import o_voxel
        except ImportError as e:
            raise RuntimeError(f"Missing dependency: {e}")

        if not image_bytes:
            raise ValueError("No image provided")

        pipeline_type = params.get("pipeline_type", "1024_cascade")
        sparse_steps  = int(params.get("sparse_steps", 12))
        shape_steps   = int(params.get("shape_steps", 12))
        tex_steps     = int(params.get("tex_steps", 12))
        seed          = int(params.get("seed", 42))
        faces         = int(params.get("faces", -1))
        texture_size  = int(params.get("texture_size", 4096))

        target_faces = faces if faces > 0 else 1_000_000

        print("[TRELLIS] Params:", params)

        self._report(progress_cb, 5, "Loading image...")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 10, "Generating...")

        stop_evt = threading.Event()

        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 85, "Generating...", stop_evt, 5.0),
                daemon=True
            )
            progress_thread.start()

        try:
            outputs = self._model.run(
                image,
                seed=seed,
                preprocess_image=True,
                pipeline_type=pipeline_type,
                sparse_structure_sampler_params={"steps": sparse_steps},
                shape_slat_sampler_params={"steps": shape_steps},
                tex_slat_sampler_params={"steps": tex_steps},
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Pipeline failed: {e}")
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=5)

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 87, "Processing mesh...")

        mesh = outputs[0]
        mesh.simplify(min(target_faces, 16_777_216))

        self._report(progress_cb, 93, "Exporting GLB...")

        glb = o_voxel.postprocess.to_glb(
            vertices          = mesh.vertices,
            faces             = mesh.faces,
            attr_volume       = mesh.attrs,
            coords            = mesh.coords,
            attr_layout       = mesh.layout,
            voxel_size        = mesh.voxel_size,
            aabb              = [[-0.5]*3, [0.5]*3],
            decimation_target = target_faces,
            texture_size      = texture_size,
            remesh            = True,
            verbose           = False,
        )

        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.glb"
        path = self.outputs_dir / filename

        glb.export(str(path), extension_webp=True)

        self._report(progress_cb, 100, "Done")
        print("[TRELLIS] Saved:", path)

        return path

    def _setup_vendor(self):
        if not _VENDOR_DIR.exists():
            raise RuntimeError("vendor/ folder missing")

        import torch

        vendor_str = str(_VENDOR_DIR)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)

    def _setup_env(self):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")
