import csv

# Open the CSV file
source_file = './data/coaching_en/SmartCoachingCalls.csv'
with open(source_file, 'r', newline='') as csvfile:
    # Create a reader object
    csv_reader = csv.reader(csvfile)
    print("Reading CSV file:", source_file)
    # print("First 5 rows:")
    # Iterate through the rows and print the first 5 rows
    # for i, row in enumerate(csv_reader):
    #     if i < 2:
    #         print(row)
    #     else:
    #         break
    # Export the first 2 rows to a new CSV file
    output_file = './data/coaching_en/SmartCoachingCalls_sample.csv'
    with open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csvfile.seek(0)  # Reset file pointer to the beginning
        for i, row in enumerate(csv_reader):
            if i < 100:
                csv_writer.writerow(row)
            else:
                break
    print("Sample exported to:", output_file)