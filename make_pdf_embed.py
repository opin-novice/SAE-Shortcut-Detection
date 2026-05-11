import os
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPORT_MD = r'd:\SAE\SAE_ablation_report.md'
OUT_PDF = r'd:\SAE\SAE_ablation_report_with_graphs.pdf'
IMAGE_DIR = r'd:\SAE\graph_res'
OTHER_IMGS = [r'd:\SAE\results\diagnostic_heatmaps.png', r'd:\SAE\results\top_activating_features.png']

def add_text_page(pdf, text, fontsize=10):
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if not line.strip():
            wrapped.append('')
        else:
            wrapped.extend(textwrap.wrap(line, width=100))
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.03, 0.98, '', va='top')
    y = 0.95
    for line in wrapped:
        fig.text(0.03, y, line, fontsize=fontsize, family='serif')
        y -= 0.013
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5,11))
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf, img_path, caption=None):
    try:
        img = plt.imread(img_path)
    except Exception as e:
        print(f"[WARN] Could not read image {img_path}: {e}")
        return
    h, w = img.shape[0], img.shape[1]
    aspect = w / h
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.8])
    ax.imshow(img)
    ax.axis('off')
    if caption:
        fig.text(0.5, 0.05, caption, ha='center', fontsize=11, family='serif')
    pdf.savefig(fig)
    plt.close(fig)


def main():
    if not os.path.exists(REPORT_MD):
        print(f"Report markdown not found: {REPORT_MD}")
        return
    with open(REPORT_MD, 'r', encoding='utf-8') as f:
        md = f.read()

    images = []
    if os.path.isdir(IMAGE_DIR):
        for fn in sorted(os.listdir(IMAGE_DIR)):
            if fn.lower().endswith('.png') or fn.lower().endswith('.jpg'):
                images.append(os.path.join(IMAGE_DIR, fn))
    for p in OTHER_IMGS:
        if os.path.exists(p):
            images.append(p)

    if not images:
        print("No images found to embed. Will produce text-only PDF.")

    print(f"Creating PDF: {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5,11))
        fig.text(0.5, 0.7, 'SAE Shortcut Detection — Comprehensive Report', ha='center', fontsize=18, family='serif', weight='bold')
        fig.text(0.5, 0.65, 'Generated via automated embedding script', ha='center', fontsize=12, family='serif')
        pdf.savefig(fig)
        plt.close(fig)

        # Add markdown text pages
        add_text_page(pdf, md, fontsize=11)

        # Add image pages
        for img in images:
            caption = os.path.basename(img)
            add_image_page(pdf, img, caption=caption)

    print(f"PDF created: {OUT_PDF}")

if __name__ == '__main__':
    main()
