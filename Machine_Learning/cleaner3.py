import csv

input_file = "dataset.csv"  # Your existing CSV file
output_file = "dataset_cleaned.csv"  # Name for the new CSV file
columns_to_keep = ['step', 'type', 'amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']  # Columns you want to keep

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    headers = reader.fieldnames

    # Filter columns to keep
    new_headers = [header for header in headers if header in columns_to_keep]

    writer = csv.DictWriter(outfile, fieldnames=new_headers)
    writer.writeheader()

    for row in reader:
        # Write rows with selected columns
        writer.writerow({col: row[col] for col in new_headers})