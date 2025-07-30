import os
import shutil
from PyPDF2 import PdfReader, PdfWriter


def split_pdf_by_pages(input_path, output_folder, max_pages=20):
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if total_pages <= max_pages:
        print(f"{os.path.basename(input_path)} has {total_pages} pages, copying without split.")
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        shutil.copy2(input_path, output_path)  # Copy the file to output folder
        print(f"Copied to: {output_path}")
        return

    # Calculate number of parts needed
    num_parts = (total_pages + max_pages - 1) // max_pages

    for part in range(num_parts):
        writer = PdfWriter()
        start = part * max_pages
        end = min(start + max_pages, total_pages)

        # Add pages for this chunk
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        output_filename = os.path.join(
            output_folder,
            f"{base_name}_part{part + 1}.pdf"
        )
        with open(output_filename, 'wb') as out_pdf:
            writer.write(out_pdf)

        print(f"Created: {output_filename} with pages {start + 1} to {end}")


def process_pdfs_in_folder(folder_path, output_folder, max_pages=20):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            input_file = os.path.join(folder_path, filename)
            split_pdf_by_pages(input_file, output_folder, max_pages=max_pages)


if __name__ == "__main__":
    input_folder = "C:\\Users\\animesh.patni\\Downloads\\Legal_Document - Copy"
    output_folder = "C:\\Users\\animesh.patni\\Downloads\\Legal_Document"

    process_pdfs_in_folder(input_folder, output_folder, max_pages=20)
