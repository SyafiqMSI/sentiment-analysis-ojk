import pandas as pd

file_path = "data/logic-exel.xlsx"  
sheet_name = "Sheet1"         
df = pd.read_excel(file_path, sheet_name=sheet_name)

page_titles = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[49, 50])

def clean_column_name(name):
    return name.replace("Sum of ", "")

def extract_page_number(page):
    return ''.join(filter(str.isdigit, str(page)))

hasil = []

for index, row in df.iterrows():
    row_label = row[0]  
    headers = []
    
    for col in df.columns[2:]: 
        value = pd.to_numeric(row[col], errors='coerce')
        if value > 0:  
            cleaned_col_name = clean_column_name(col)
            matching_page = page_titles[page_titles.iloc[:, 1].str.contains(cleaned_col_name, case=False, na=False)]
            if not matching_page.empty:
                page = matching_page.iloc[0, 0]  
                page_number = extract_page_number(page)  
                headers.append(f"{cleaned_col_name} {page_number}")  
    
    if headers:
        hasil.append(f"{row_label}\n" + "\n".join(headers))  

output_txt_path = "data/hasil/hasil_output.txt"
with open(output_txt_path, "w") as f:
    f.write("\n\n".join(hasil))  
print(f"Hasil disimpan di {output_txt_path}")