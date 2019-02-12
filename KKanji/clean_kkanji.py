import os
#base_dir = "\kkanji2\U+4C61\e6eeb23552e1a21c.png"
base_dir = "kkanji2"

for dirName, subdirList, fileList in os.walk(base_dir):
    print(dirName[10:]) # labels
    for fname in fileList:
         print('\t%s -> %s' % (open(dirName + "\\" +  fname), dirName[10:])) # actual files for data