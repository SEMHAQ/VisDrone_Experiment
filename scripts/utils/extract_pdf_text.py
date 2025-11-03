import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/utils/extract_pdf_text.py <input_pdf> <output_txt>")
        sys.exit(1)

    input_pdf = Path(sys.argv[1])
    output_txt = Path(sys.argv[2])

    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print("Missing dependency: pdfminer.six. Install with 'pip install pdfminer.six'.")
        sys.exit(1)

    if not input_pdf.exists():
        print(f"Input PDF not found: {input_pdf}")
        sys.exit(1)

    text = extract_text(str(input_pdf))
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(text, encoding='utf-8')
    print(f"✅ Extracted text to {output_txt} (chars: {len(text)})")

if __name__ == "__main__":
    main()