import os

files = os.listdir()
for file in files:
    if file.endswith("two.out"):
        theFile = open(file, 'r')
        lines = theFile.readlines()
        lastLine = lines[-1]
        secondLastLine = lines[-2]
        if 'end of ' not in lastLine:
            print(file)
            print(secondLastLine, lastLine)
        theFile.close()
