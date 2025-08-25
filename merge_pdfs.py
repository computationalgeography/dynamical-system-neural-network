from pypdf import PdfWriter, PdfReader

figure_directory = "../figures/"

first_pdf = figure_directory + "land_art_one/r2_by_variable.pdf"
second_pdf = figure_directory + "land_art_two/r2_by_variable.pdf"
#first_pdf = figure_directory + "land_obs_one/r2_by_variable.pdf"
#second_pdf = figure_directory + "land_obs_two/r2_by_variable.pdf"

writer = PdfWriter(clone_from = first_pdf)
second_page = PdfReader(second_pdf).pages[0]

for page in writer.pages:
    page.merge_page(second_page, over=True)

writer.write(figure_directory + "merged/" + "r2_by_variable.pdf")

#stamp = PdfReader("overlay.pdf").pages[0]
#writer = PdfWriter(clone_from="form_1099_nec_page_4.pdf")
#for page in writer.pages:
#    page.merge_page(stamp, over=True)
#
#writer.write("merged.pdf")
