from pypdf import PdfWriter, PdfReader

first_pdf = "/Users/karss101/tmp/r2_by_scenario.pdf"
second_pdf = "/Users/karss101/misc/research/models/dyn_sys_nn/figures/r2_by_scenario.pdf"

writer = PdfWriter(clone_from = first_pdf)
second_page = PdfReader(second_pdf).pages[0]

for page in writer.pages:
    page.merge_page(second_page, over=True)

writer.write("test.pdf")

#stamp = PdfReader("overlay.pdf").pages[0]
#writer = PdfWriter(clone_from="form_1099_nec_page_4.pdf")
#for page in writer.pages:
#    page.merge_page(stamp, over=True)
#
#writer.write("merged.pdf")
