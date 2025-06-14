#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --nemo-ref)
        NEMO_REF="$2"
        shift 2
        ;;
    --mcore-ref)
        MCORE_REF="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 --nemo-ref NEMO_REF --mcore-ref MCORE_REF"
        exit 1
        ;;
    esac
done

# Check if required arguments are provided
if [ -z "$NEMO_REF" ] || [ -z "$MCORE_REF" ]; then
    echo "Error: --nemo-ref and --mcore-ref are required"
    echo "Usage: $0 --nemo-ref NEMO_REF --mcore-ref MCORE_REF"
    exit 1
fi

# Nemo-run has conflicting dependencies to export-deploy:
# They collide on nvidia-pytriton (export-deploy) and torchx (nemo-run)
# via urllib3.
uv pip install --no-cache-dir --upgrade nemo-toolkit[automodel,common-only,nlp-only,eval,multimodal-only]@git+https://github.com/NVIDIA/NeMo.git@${NEMO_REF}

# megatron-core and export-deploy are dependencies, but for development
# we override with latest VCS commits
uv pip uninstall megatron-core
uv pip install --no-cache-dir --upgrade \
    "numpy<2.0.0" "megatron_core@git+https://github.com/NVIDIA/Megatron-LM.git@${MCORE_REF}"
