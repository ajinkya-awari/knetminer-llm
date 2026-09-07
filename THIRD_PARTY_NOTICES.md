# Third-Party Notices

[README](README.md) | [Architecture](docs/architecture.md) | [Data and safety](docs/data-and-safety.md)

This repository depends on third-party Python packages. Their licenses remain with their respective
copyright holders and are available from the projects linked in `pyproject.toml` or their installed
package metadata.

The benchmark is designed for [Open Targets Platform](https://platform.opentargets.org/) release
26.06 data, which Open Targets publishes under CC0. No Open Targets source snapshot is included in
this source release. The static demonstration includes only the validated, derived stable-ID and
forward-edge inventory described in the README.

The optional local synthesis adapter targets
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Model files are not included in this repository. Users
must review and comply with the model repository's license before obtaining it separately.

The optional HGT benchmark encodes node labels with
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, distributed under Apache-2.0. Model files are not
included in this repository.

[KnetMiner](https://knetminer.com/) is referenced only to identify the benchmark's research context.
This independent project is not affiliated with, endorsed by, or an implementation of the KnetMiner
service.

These notices document attribution boundaries; they do not replace the license files or usage terms
published by each dependency or model provider.
