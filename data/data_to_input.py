import glob
import os
import pdftotext
import nltk
import math
from nltk.tokenize import RegexpTokenizer

## change "annual-reports" to the folder of your choice
annual_reports = glob.glob(os.path.join(os.path.dirname(__file__), "annual-reports") + "/*")
full_text_list = []
banks = []
tokenizer = RegexpTokenizer('\s+', gaps=True)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for ar_pdf in annual_reports:
    # banks.append(ar_pdf[ar_pdf.rfind('/')+1:ar_pdf.rfind('-')])
    try:
        with open(ar_pdf, "rb") as f:
            pdf = pdftotext.PDF(f)
            full_string = "\n\n".join(pdf)
            divide_by =  math.ceil(len(full_string) / 5e4)
            full_text = tokenizer.tokenize(full_string)
            if divide_by > 1:
                full_text_split = list(chunks(full_text, len(full_text) // divide_by))
            else:
                full_text_split = [full_text]

            for text_split in full_text_split:
                banks.append(ar_pdf[ar_pdf.rfind('/')+1:])
                full_text_list.append(" ".join(text_split))
    except Exception as err:
        print(f"Error processing file: {ar_pdf}")
        print(err.args)

assert len(full_text_list) == len(banks)

with open("input/documents_ar.txt", "w") as fout_ar:
    with open("input/document_ids_ar.txt", "w") as fout_bank:
        for ar_text, bank in zip(full_text_list, banks):
            if len(ar_text) > 1000:
                fout_ar.write(ar_text)
                fout_ar.write("\n")
                fout_bank.write(bank)
                fout_bank.write("\n")
            else:
                continue
