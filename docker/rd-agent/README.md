# RD-Agent Sandbox

This directory defines the high-risk execution sandbox contract for Auto-Research V3.

- Host read mount: `sandbox/rd-agent/in` -> `/workspace/in`
- Host write mount: `sandbox/rd-agent/out` -> `/workspace/out`
- No repo root, vault, `.env`, or `.runtime` mounts

The current `runner.py` is a scaffold entrypoint so the orchestration layer can be tested before the full RD-Agent image is enabled.
