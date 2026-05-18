# Derived image: fluorosim + PyTorch.
# PyTorch is an *optional* fluorosim dep — used by the Slang autodiff path that
# wraps the renderer as a torch.autograd.Function. Required for Phase 5
# (gradient-based pose registration via register_phantom.py).
#
# Build:
#   docker build -t fluorosim-torch -f bridge/fluorosim_torch.Dockerfile bridge/
#
# Run via bridge/run_register.sh (uses the fluorosim-torch tag).

FROM fluorosim

# CUDA 12.6 runtime in the base; install the CUDA-12.1 PyTorch wheel (it's
# smaller than the cu124 wheel and binary-compatible with the 12.6 runtime,
# which matters for unreliable links).  --break-system-packages because the
# base image's Python is managed by the OS package manager (PEP 668).
# Long timeout + retries because the wheels are ~1 GB.
RUN pip install --no-cache-dir --break-system-packages \
        --timeout 600 --retries 10 \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1
