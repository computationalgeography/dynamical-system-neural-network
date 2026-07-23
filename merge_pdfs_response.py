from pypdf import PdfWriter, PdfReader
import sys

scenario = "land_obs_one"

figure_directory = "../figures/" + scenario + "/"

output_sub_directory = "../figures/merged/"

# 247 35  528 534 535 565 68  815 818
def merge_pdfs(file_name):
    first_pdf = figure_directory + "247/" + file_name
    writer = PdfWriter(clone_from = first_pdf)

    ids = ["35",  "528", "534", "535", "565", "68",  "815", "818"]
    for id in ids:
        second_pdf = figure_directory + id + "/" + file_name
        second_page = PdfReader(second_pdf).pages[0]
    
        for page in writer.pages:
            page.merge_page(second_page, over=True)

    writer.write(output_sub_directory + scenario + "_" + file_name)

merge_pdfs("response.pdf")
