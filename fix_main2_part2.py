import re

with open('main2.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Make it explicit that 18 is the number of transactions, but 9 is the number of arrays.
content = content.replace(r'approximately \times4=72$\,bytes in single precision', r'approximately 18 redundant memory transactions (\times4=72$\,bytes) in single precision')

with open('main2.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done.")
