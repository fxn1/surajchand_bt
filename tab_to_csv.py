import csv
import os
from datetime import datetime

input_file = "spx_2023.txt"
output_file = "output.csv"
monthly_dir = "monthly_csv"
os.makedirs(monthly_dir, exist_ok=True)
monthly_data = {}

lines=0
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)

    for line in infile:
        # Split on any whitespace (tabs or multiple spaces)
        parts = line.strip().split()

        if parts[0].startswith("202"):
            parts = [" ".join(parts[:2])] + parts[2:]
        writer.writerow(parts)
        lines += 1

        # Join the date + time back together (first two tokens)
        if parts[0].startswith("202"):
            dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            ym = dt.strftime("%Y-%m")
            if ym not in monthly_data:
                monthly_data[ym] = []
            monthly_data[ym].append(parts)

# --- NEW: write monthly CSV files ---
months=0
for ym, rows in monthly_data.items():
    out_path = os.path.join(monthly_dir, f"{ym}.csv")
    months+=1
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        writer.writerows(rows)

print("Conversion complete lines=", lines, "months=", months)
