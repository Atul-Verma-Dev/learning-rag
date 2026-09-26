from pypdf import PdfReader

d = {}

reader = PdfReader("ED_Exp_7.pdf")
for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    d[page_num+1] = text


print(d)
