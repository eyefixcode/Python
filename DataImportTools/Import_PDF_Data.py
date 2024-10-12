#import pdf file

from PyPDF2 import PdfFileReader
file = open("file.pdf", "rb")
content = PdfFileReader(file)
text = content.getPage(0)
text.extractText()
''
content.numpages