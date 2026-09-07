# README Assets

[Project README](../../README.md) | [Architecture](../architecture.md) |
[Evidence workbench](../demo/)

`knetminer-evidence-flow.svg` is a hand-authored, dependency-free SVG for the repository README. It
uses only local vector elements and embedded CSS. The one-shot animation reveals the existing graph
and validation pipeline; it does not render benchmark values or generated biological claims.

The asset has no JavaScript, remote font, external image, tracking request, or embedded raster data.
Its `prefers-reduced-motion` rule disables animation while preserving the complete static frame.

Validate it from the repository root with:

```powershell
[xml](Get-Content -Raw docs/assets/knetminer-evidence-flow.svg) | Out-Null
Select-String -Path docs/assets/knetminer-evidence-flow.svg -Pattern `
  '<script', '(?:href|src)="https?://', 'on(?:load|click|error)=', 'data:image'
```

The XML command must exit 0 and the scan must return no matches.

GitHub clients that disable CSS animation still receive the complete static frame. The asset is
decorative documentation only and is not a benchmark chart, runtime component, or evidence source.
