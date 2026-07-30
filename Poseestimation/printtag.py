"""
Generates a single PDF containing all 6 AprilTags, each printed at an exact
physical size, with its tag ID/name labeled underneath.

Print this PDF at 100% / "Actual Size" -- do NOT use "Fit to Page".
After printing, measure each tag with calipers to confirm actual size
(see TAG_SIZE_CM below for the intended target).
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import os

# --- Configuration ---
TAG_SIZE_CM = 1.0          # target physical edge length of each tag's black square
LABEL_FONT_SIZE = 10      # pt
SPACING_CM = 5.0           # extra room below/around each tag for its label + margin
COLS = 3                   # tags per row
MARGIN_CM = 2.0            # page margin

# tag_id -> (filename, display name / face label)
TAGS = {
    0: ("tag36h11_00.png", "ID 0 - Top (+Z)"),
    1: ("tag36h11_01.png", "ID 1 - Right (+X)"),
    2: ("tag36h11_02.png", "ID 2 - Left (-X)"),
    3: ("tag36h11_03.png", "ID 3 - Front (+Y)"),
    4: ("tag36h11_04.png", "ID 4 - Back (-Y)"),
    5: ("tag36h11_05.png", "ID 5 - Bottom (-Z)"),
}

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # always the same folder this script is in
OUTPUT_PDF = os.path.join(INPUT_DIR, "all_tags_1cm.pdf")


def build_pdf():
    resolved_input_dir = os.path.abspath(INPUT_DIR)
    print(f"Looking for tag PNGs in: {resolved_input_dir}")
    if os.path.isdir(resolved_input_dir):
        existing_files = os.listdir(resolved_input_dir)
        print(f"Files found in that folder: {existing_files if existing_files else '(empty)'}")
    else:
        print(f"WARNING: '{resolved_input_dir}' is not a valid directory.")

    c = canvas.Canvas(OUTPUT_PDF, pagesize=A4)
    page_width, page_height = A4

    cell_w = TAG_SIZE_CM * cm + SPACING_CM * cm
    cell_h = TAG_SIZE_CM * cm + SPACING_CM * cm

    x_start = MARGIN_CM * cm
    y_start = page_height - MARGIN_CM * cm

    col = 0
    row = 0
    missing = []

    for tag_id, (filename, label) in sorted(TAGS.items()):
        filepath = os.path.join(INPUT_DIR, filename)

        x = x_start + col * cell_w
        y = y_start - row * cell_h - (TAG_SIZE_CM * cm)

        if not os.path.exists(filepath):
            missing.append(filename)
            # Draw a placeholder box + warning text so the layout stays visible
            c.rect(x, y, TAG_SIZE_CM * cm, TAG_SIZE_CM * cm)
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + (TAG_SIZE_CM * cm) / 2, y + (TAG_SIZE_CM * cm) / 2,
                                 "MISSING")
        else:
            # drawImage at an exact width/height in cm -- this is what guarantees
            # physical size regardless of the source PNG's pixel dimensions or
            # embedded DPI metadata.
            c.drawImage(filepath, x, y, width=TAG_SIZE_CM * cm, height=TAG_SIZE_CM * cm)

        # Label directly below the tag
        c.setFont("Helvetica", LABEL_FONT_SIZE)
        c.drawCentredString(x + (TAG_SIZE_CM * cm) / 2, y - 0.5 * cm, label)

        col += 1
        if col >= COLS:
            col = 0
            row += 1

        # Move to a new page if we run out of vertical room
        if y_start - (row + 1) * cell_h < MARGIN_CM * cm:
            c.showPage()
            row = 0
            col = 0

    c.save()

    print(f"Saved: {OUTPUT_PDF}")
    print(f"Each tag placed at exactly {TAG_SIZE_CM} cm x {TAG_SIZE_CM} cm.")
    print("Print at 100% / 'Actual Size' -- NOT 'Fit to Page'.")
    if missing:
        print("\nWARNING: the following tag files were not found and were left blank:")
        for m in missing:
            print(f"  - {m}")
        print(f"Place these files in '{INPUT_DIR}' and re-run to fill them in.")


if __name__ == "__main__":
    build_pdf()