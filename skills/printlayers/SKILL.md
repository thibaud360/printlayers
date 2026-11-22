---
name: printlayers
description: Split a photo or drawing into 2-4 flat ink layers for printmaking: linocut (mirrored, ready to carve), screen print, stencil with bridges (spray, Cricut) or riso halftone. Use when the user wants to print, carve or cut their own image, or types /printlayers.
---

# printlayers

Turns the user's own photo or drawing into flat ink layers with `scripts/printlayers.py` (Python 3, Pillow, NumPy, OpenCV, SciPy, scikit-learn, scikit-image). Pure code, no image model. Black in every SVG = ink.

## Modes

| Mode | What you get |
|---|---|
| `linocut` | One bold black/white layer with optional carved-line texture in the midtones, plus a horizontally mirrored SVG/PNG for carving |
| `screenprint` | 2-4 colour layers by colour separation, light inks first, registration marks, small trap so layers overlap a hair |
| `stencil` | Layers for spray paint or a Cricut. Every island that would fall out gets a thin bridge. Stencil layers are cumulative (lightest ink first, darker ones on top) |
| `riso` | Two inks picked from a risograph palette, each with its own halftone screen angle |

## Steps

1. Find the image. Ask nothing if the mode is clear; if it isn't, pick from what they say: carve / lino / block -> `linocut`; screen / poster / silkscreen -> `screenprint`; spray / stencil / Cricut / cut -> `stencil`; riso / halftone / zine -> `riso`.
2. `pip install pillow numpy opencv-python scipy scikit-learn scikit-image` (add `--break-system-packages` if pip asks).
3. Run: `python scripts/printlayers.py INPUT --mode MODE -o printlayers_out`
4. Look at `*_proof.png` and `*_sheet.png` before sending. If the result is too busy, rerun with `--detail 2`; if too coarse, `--detail 4`. Photos with flat dark backgrounds work best; very low-contrast photos may need a brighter copy first.
5. Send the proof, the sheet and the SVGs. Say which layer to print or spray first, and for linocut say that the `_mirror` files are the ones to transfer onto the block.

## Options

- `--layers N` screenprint 2-4 (default 3), stencil 1-4 (default 2)
- `--inks "#hex" ...` ink colours (default: picked from the photo; riso: best pair from a riso palette)
- `--detail 1-5` 1 = bold and simple, 5 = fine (default 3)
- `--width-mm N` physical width of the print, sets the SVG size and the minimum feature size (default 150)
- `--paper "#hex"` paper colour in the proof
- `--texture lines|none`, `--line-mm`, `--angle` linocut carved lines (`none` gives plain black/white)
- `--bridge-mm` stencil bridge width (default 0.9); `--cell-mm` riso dot spacing (default 1.5)
- `--mirror on` also write mirrored files for other modes (relief printing)
- `--misreg 0` perfect registration in the proof; `--no-marks` no registration marks; `--px` working width; `--strip` also write an original | proof | layers strip

## Outputs

For `photo.jpg` in mode `linocut`, in the output folder: `photo_linocut_layer1.svg` and `.png`, `photo_linocut_layer1_mirror.svg` and `.png`, `photo_linocut_proof.png`, `photo_linocut_sheet.png`. Multi-layer modes write `layer1`, `layer2`, ... SVGs include registration marks in a 6 mm margin.

## Notes

- Cricut / Silhouette: import the SVG, the size is already in mm. Black shapes are the holes to cut (where paint goes).
- Layers print light to dark. The proof overprints them on paper with slight misregistration, so it looks like a real print.
- It's a prep tool for the user's own image. Don't describe the result as AI art.

## If the script is missing

`git clone --depth 1 https://github.com/thibaud360/printlayers /tmp/printlayers` and use `/tmp/printlayers/skills/printlayers/scripts/printlayers.py`.
