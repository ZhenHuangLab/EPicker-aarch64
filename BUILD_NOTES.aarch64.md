# EPicker — Linux aarch64 + CUDA 13 + GB10 (sm_121)

Upstream: https://github.com/thuem/EPicker (CenterNet-based continual-learning picker)

## Result: 🟢 backbone PASS — 2 structural fixes, both x86/legacy-torch problems

The table's verdict was `python==3.6.11 too old, torch==1.7.1+cu110 too old,
neither conda nor pip can install it`. The two things that actually made EPicker
architecture-locked are:

1. `lib/external/nms.cpython-37m-x86_64-linux-gnu.so` — a **prebuilt x86-64
   CPython-3.7 binary** checked into the repo.
2. `lib/models/networks/DCNv2/` — a bundled CUDA/C++ deformable-convolution
   extension written against **THC** (`THC/THC.h`, `THCudaMalloc`,
   `THCudaBlas_SgemmBatched`, `THCState`). THC was deleted from PyTorch in 1.11,
   so this extension cannot build against *any* modern PyTorch on *any*
   architecture — 40 THC references across 6 source files.

## Fix 1 — replace the DCNv2 CUDA extension with `torchvision.ops.deform_conv2d`

`torchvision.ops.deform_conv2d` **is** modulated deformable convolution v2 (Zhu
et al. 2019) with upstream-maintained CPU and CUDA kernels. Rather than port 40
THC call sites, `lib/models/networks/DCNv2/dcn_v2.py` was rewritten as a thin
wrapper around it and the native build was deleted entirely
(`src/`, `setup.py`, `make.sh`, `testcpu.py`, `testcuda.py`, `dcn_v2_onnx.py`).

Only `DCN` is kept — it is the sole symbol EPicker imports
(`pose_dla_dcn.py`, `resnet_dcn.py`). The explicit-offset `DCNv2` class and the
`DCNv2Pooling` / `DCNPooling` ROI variants were referenced only by the deleted
test scripts.

Numerical check — with zero-initialised offsets and `sigmoid(0) = 0.5`
modulation, DCNv2 must reduce to `0.5 x` a plain convolution:

| device | `max |DCN(x) - 0.5*conv2d(x)|` |
|---|---|
| cpu | 4.768e-07 |
| cuda | 4.768e-07 |

Gradients flow to weights and input, and the layer still runs with non-zero
learned offsets.

This removes the per-architecture / per-PyTorch-version native build for good.

## Fix 2 — rebuild the Cython `nms` module (and un-break it for NumPy 2)

```bash
cd lib/external
rm -f nms.cpython-37m-x86_64-linux-gnu.so nms.c
python setup.py build_ext --inplace && rm -rf build
```

`nms.pyx` first needs three NumPy-2 fixes — `np.int` / `np.float` aliases were
removed and `np.int_t` no longer exists in Cython's NumPy declarations
(`nms.pyx:32:23: Invalid type.`):

```cython
def nms(np.ndarray[np.float32_t, ndim=2] dets, float thresh):          # was np.float thresh
    ...
    cdef np.ndarray[np.int64_t, ndim=1] order = scores.argsort()[::-1].astype(np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int64)                          # was dtype=np.int
```

Produces `nms.cpython-311-aarch64-linux-gnu.so`; `nms.nms` and `nms.soft_nms`
both return correct suppression results.

## Environment

```bash
mamba create -y -p ./env -c conda-forge python=3.11 numpy scipy matplotlib \
      pillow opencv cython numba pycocotools
./env/bin/pip install torch torchvision progress mrcfile
```

Resolved: torch 2.13.0+cu130, torchvision 0.28.0+cu130 — i.e. torch **2.13**
instead of the pinned 1.7.1, with no code changes beyond the two fixes above.

## Smoke test — CenterNet DLA-34 backbone

```python
from models.model import create_model
m = create_model('dla_34', {'hm':1,'wh':2,'reg':2}, 256, 'model/dla34-ba72cf86.pth').cuda()
```

| check | result |
|---|---|
| parameters | 20.17 M |
| forward (1x3x512x512) | heads `hm (1,1,128,128)`, `wh (1,2,128,128)`, `reg (1,2,128,128)` + feature maps |
| finite | all outputs |
| backward | 227 parameter tensors received gradients, all finite |

The bundled ImageNet backbone `model/dla34-ba72cf86.pth` loads correctly.

A full picking run needs real micrographs plus a trained EPicker checkpoint,
which are not distributed with the repository; the architecture-blocking parts
are resolved and verified.
