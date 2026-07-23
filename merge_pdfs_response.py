from pypdf import PdfWriter, PdfReader
import sys

figure_directory = "../figures/"

run = sys.argv[1]


if run == 'obs':
    output_sub_directory = "land_obs"
else:
    output_sub_directory = "land_art"

def merge_pdfs(file_name):
    first_pdf = figure_directory + output_sub_directory + "_one" + "/" + file_name
    second_pdf = figure_directory + output_sub_directory + "_two" + "/" + file_name

    writer = PdfWriter(clone_from = first_pdf)
    second_page = PdfReader(second_pdf).pages[0]

    for page in writer.pages:
        page.merge_page(second_page, over=True)

    writer.write(figure_directory + "merged/" + output_sub_directory + "/" + file_name)

merge_pdfs("r2_by_variable.pdf")
merge_pdfs("tss_modartcomp_best_fit_only_fit_thr.pdf")

#first_pdf = figure_directory + output_sub_directory + "_one" + "/r2_by_variable.pdf"
#second_pdf = figure_directory + output_sub_directory + "_two" + "/r2_by_variable.pdf"
#
#writer = PdfWriter(clone_from = first_pdf)
#second_page = PdfReader(second_pdf).pages[0]
#
#for page in writer.pages:
#    page.merge_page(second_page, over=True)
#
#writer.write(figure_directory + "merged/" + output_sub_directory + "/" + "r2_by_variable.pdf")

# tss





#first_pdf = figure_directory + "land_obs_one/tss_modartcomp_best_fit_only_fit_thr.pdf"
#second_pdf = figure_directory + "land_obs_two/tss_modartcomp_best_fit_only_fit_thr.pdf"
#writer.write(figure_directory + "merged/" + "tss_modartcomp_best_fit_only_fit_thr.pdf")




#stamp = PdfReader("overlay.pdf").pages[0]
#writer = PdfWriter(clone_from="form_1099_nec_page_4.pdf")
#for page in writer.pages:
#    page.merge_page(stamp, over=True)
#
#writer.write("merged.pdf")
