#import xml file

from platform import python_branch
import xml.etree.ElementTree as ET

content = ET.parse("file.xml")
root = content.getroot()
for child in root:
    print(child.tag, child.text)
    

#xmlcode
#<root>
#    <key>Python</key>
#    <value>Learning</value>
#</root>    

#return
#key Python
#value Learning