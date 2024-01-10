#import csv contents

import csv
with open("file.csv") as file:
    content = csv.reader(file)
    for row in content:
        print(row)
        
        