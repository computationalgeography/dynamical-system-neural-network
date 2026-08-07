from pypdf import PdfWriter, PdfReader
import sys



output_sub_directory = "../figures/merged/"

# 247 35  528 534 535 565 68  815 818
def merge_pdfs(file_name, scenario):
    figure_directory = "../figures/" + scenario + "/"
    first_pdf = figure_directory + "247/" + file_name
    writer = PdfWriter(clone_from = first_pdf)

    ids = ["35",  "528", "534", "535", "565", "68",  "815", "818"]
    for id in ids:
        second_pdf = figure_directory + id + "/" + file_name
        second_page = PdfReader(second_pdf).pages[0]
    
        for page in writer.pages:
            page.merge_page(second_page, over=True)

    writer.write(output_sub_directory + scenario + "_" + file_name)

scenario = "land_obs_one"
merge_pdfs("response.pdf", scenario)

scenario = "land_obs_two"
merge_pdfs("response.pdf", scenario)
