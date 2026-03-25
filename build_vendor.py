"""
Build the vendor/ directory for the TRELLIS.2 extension.

Run this script once (with the app's venv active) to populate vendor/.
The resulting vendor/ folder is committed to the extension repository
so end users never need to install anything at runtime.

Usage:
    python build_vendor.py

Requirements (must be run from the app's venv):
    - pip (always available)
    - PyTorch + CUDA (must be available at inference time anyway)

The following compiled CUDA extensions must already be installed in the
app's venv — they cannot be vendored as pure Python:

    pip install o-voxel
    pip install nvdiffrast            # from NVlabs
    pip install git+https://github.com/JeffreyXiang/nvdiffrec.git
    pip install cumesh
    pip install flexgemm
    pip install flash-attn            # or: pip install xformers (fallback)
"""

import io
import subprocess
import sys
import zipfile
from pathlib import Path

VENDOR       = Path(__file__).parent / "vendor"
TRELLIS2_ZIP = "https://github.com/microsoft/TRELLIS.2/archive/refs/heads/main.zip"

# Pure-Python packages to vendor (no compilation needed)
PURE_PACKAGES = [
    "easydict",   # configuration dict used internally by trellis2
    "plyfile",    # PLY mesh format I/O
    "einops",     # tensor reshaping helpers
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list, **kwargs):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def vendor_pure_package(package: str, dest: Path) -> None:
    """Install a pure-Python package into vendor/ via pip --target.

    Using pip --target handles all package layouts (flat, src/, namespace
    packages) and always includes the .dist-info directory needed by
    importlib.metadata.
    """
    run([sys.executable, "-m", "pip", "install",
         "--no-deps",
         "--target", str(dest),
         "--upgrade",
         package])
    print(f"  Vendored {package}.")


def vendor_trellis2(dest: Path) -> None:
    """Download TRELLIS.2 source and extract only the trellis2/ package into vendor/."""
    import urllib.request

    trellis2_dest = dest / "trellis2"
    if trellis2_dest.exists():
        print("  trellis2/ already present, skipping.")
        return

    print("  Downloading TRELLIS.2 source from GitHub...")
    with urllib.request.urlopen(TRELLIS2_ZIP, timeout=180) as resp:
        data = resp.read()

    # The ZIP root folder is "TRELLIS.2-main/" (GitHub archive naming)
    prefix = "TRELLIS.2-main/trellis2/"
    strip  = "TRELLIS.2-main/"

    extracted = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel    = member[len(strip):]
            target = dest / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted += 1

    if extracted == 0:
        raise RuntimeError(
            f"No files were extracted from the ZIP. "
            f"The expected prefix '{prefix}' was not found.\n"
            "Check that the GitHub archive structure matches and update the "
            "'prefix' variable in vendor_trellis2() if needed."
        )

    print(f"  trellis2/ extracted to {dest} ({extracted} files).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Guard: torch must be importable — ensures we're in the right venv.
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "torch is not importable from this Python environment.\n"
            "Run build_vendor.py using the app's embedded Python (the one with PyTorch),\n"
            f"not the system Python.\nCurrent interpreter: {sys.executable}"
        )

    print(f"Building vendor/ in {VENDOR}")
    VENDOR.mkdir(parents=True, exist_ok=True)

    # 1. Pure-Python packages
    for pkg in PURE_PACKAGES:
        print(f"\n[1] Vendoring {pkg}...")
        vendor_pure_package(pkg, VENDOR)

    # 2. TRELLIS.2 source
    print("\n[2] Vendoring trellis2 source...")
    vendor_trellis2(VENDOR)

    print("\nDone! vendor/ is ready.")
    print("Commit the vendor/ directory to the extension repository.")
    print("End users will never need to install anything.")
    print()
    print("Reminder: the following compiled CUDA extensions must be in the app's venv:")
    print("  pip install o-voxel")
    print("  pip install nvdiffrast")
    print("  pip install git+https://github.com/JeffreyXiang/nvdiffrec.git")
    print("  pip install cumesh flexgemm")
    print("  pip install flash-attn   # or: pip install xformers")


if __name__ == "__main__":
    main()
