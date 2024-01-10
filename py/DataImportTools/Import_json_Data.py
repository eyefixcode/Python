#import json file

import json
with open("file.json") as file:
    content = json.load(file)
    
content.get("var1")
content.get("var2")

print(content.get("n"))
