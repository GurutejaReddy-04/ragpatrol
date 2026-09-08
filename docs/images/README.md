# RAGPatrol Documentation Assets

This directory contains visual documentation artifacts and screenshot previews displayed in the project's root `README.md`.

## Included Assets

| File | Description | Max Size |
| :--- | :--- | :--- |
| `terminal.png` | Terminal test run showing 86/86 automated tests passing | < 1 MB |
| `comparison-table.png` | Side-by-side configuration experiment comparison card | < 1 MB |
| `report-html.png` | Standalone HTML evaluation report KPI and category card | < 1 MB |
| `stub-app-run.png` | CLI output demonstrating generality against independent stub API | < 1 MB |

## How to Regenerate Assets

To update or regenerate all image assets programmatically:

```bash
# Ensure PIL is installed in your active virtual environment
python scripts/generate_docs_assets.py
```

### Manual Capture Guidelines
If capturing live application screenshots manually:
1. **Resolution & DPI:** Use 2x scaling (high-DPI) or 950px+ width.
2. **Theme:** Dark mode (slate-900 / dark terminal background).
3. **Format:** Lossless PNG with PNG optimization enabled.
4. **File Size Limit:** Keep all assets under 1 MB.
5. **Privacy:** Ensure zero personal paths (`C:\Users\...`), tokens, or production API keys are visible.
