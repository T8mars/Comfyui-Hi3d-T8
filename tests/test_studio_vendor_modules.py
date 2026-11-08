import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_studio_vendor_modules_load_and_decode_hdr(tmp_path):
    """Load the shipped module graph; syntax-only checks miss absent imports."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the actual JavaScript module check")
    vendor = Path(__file__).resolve().parents[1] / "hi3d_workbench/web/vendor/three"
    manifest = json.loads((vendor / "manifest.json").read_text(encoding="utf-8"))
    package = tmp_path / "node_modules/three"
    for name, digest in manifest["files"].items():
        data = (vendor / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (package / "package.json").write_text(
        json.dumps({"name": "three", "type": "module", "exports": "./three.module.js"}),
        encoding="utf-8",
    )
    script = tmp_path / "load-studio-modules.mjs"
    script.write_text(
        "import assert from 'node:assert/strict';\n"
        "import {FloatType} from 'three';\n"
        "import {GLTFLoader} from './node_modules/three/loaders/GLTFLoader.js';\n"
        "import {RGBELoader} from './node_modules/three/loaders/RGBELoader.js';\n"
        "import {OrbitControls} from './node_modules/three/controls/OrbitControls.js';\n"
        "import {TransformControls} from './node_modules/three/controls/TransformControls.js';\n"
        "assert.equal(typeof GLTFLoader, 'function');\n"
        "assert.equal(typeof OrbitControls, 'function');\n"
        "assert.equal(typeof TransformControls, 'function');\n"
        "const header=Buffer.from('#?RADIANCE\\nFORMAT=32-bit_rle_rgbe\\n\\n-Y 1 +X 1\\n');\n"
        "const bytes=Buffer.concat([header,Buffer.from([128,64,32,129])]);\n"
        "const image=new RGBELoader().setDataType(FloatType).parse(\n"
        "  bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));\n"
        "assert.equal(image.width,1); assert.equal(image.height,1);\n"
        "for(let i=0;i<3;i++) assert.ok(Math.abs(image.data[i]-[128,64,32][i]*2/255)<1e-6);\n"
        "assert.equal(image.data[3],1);\n",
        encoding="utf-8",
    )
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
