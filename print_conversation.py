import csv

with open("aaaedit.csv", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 4:
            print(row[2], ":",row[3])  # Fourth element (0-based indexing)