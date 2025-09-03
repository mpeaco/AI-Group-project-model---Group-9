import re

#Input - > SVG image from potrace
#Output - > Series of individual paths

#potrace only outputs relative coordinates, which will not work for our purposes.
#as such, i need to transform them into absolute coordinates.
#in order to pass to the pathfinder AI, the SVG then needs to be split into individual lines

# relativeToAbsolute:
# takes: file address
# outputs: list - contains an svg file, abstracted as

def operateHeader(header):
    h2 = header.split("\n")
    new_lines = []

    for line in h2:
        
        if 'fill="' in line:
            start = line.index('fill="') + len('fill="')
            end = line.index('"', start)
            line = line[:start] + "none" + line[end:]

        if 'stroke="' in line:
            start = line.index('stroke="') + len('stroke="')
            end = line.index('"', start)
            line = line[:start] + "#000000\" stroke-width=\"10" + line[end:]

        new_lines.append(line)

    return "\n".join(new_lines)


def operatem(pos, currentPos):
    # resolves the "m" instruction
    output = []
    output2 = str()
    output.append(pos[0]+currentPos[0])
    output.append(pos[1]+currentPos[1])
    
    for point in output:
        output2 += " " + str(point)
    
    return output2, output

def operatec(arc, currentPos):
    outputArc = []
    outputArc2 = str()
    for x in range(0,6,2):
        outputArc.append(arc[x]+currentPos[0])
        outputArc.append(arc[x+1]+currentPos[1])
    
    for point in outputArc:
        outputArc2 += " " + str(point)
    
    return outputArc2, [outputArc[4], outputArc[5]]

def operatez(current, stored):
    return 

def operatel(line, currentPos):
    outputLine = []
    outputLine2 = str()
    outputLine.append(line[0]+currentPos[0])
    outputLine.append(line[0+1]+currentPos[1])
    
    for point in outputLine:
        outputLine2 += " " + str(point)
    
    return outputLine2, outputLine

def relativeToAbsolute(file):
    output = []
    with open(file, "r") as file:
        svgData = file.read()

    header, *data2 = svgData.split("<path d")
    header = operateHeader(header)
    output.append(header)
    
    svgData = svgData.replace("\n", " ").replace("\r", " ")

    data2 = svgData.split("d=")
    data2.pop(0)

    splitters = ["MmCcLlQqAaZzTtHhVvSs"]

    currentPosition = [0,0]
    for segment in data2:        
        for line in re.split(r"(?=[MmCcLlQqAaZzTtHhVvSs])", segment):
            if line[0] == "M":
                temp = line.split(" ")
                temp[0] =float(temp[0][1:])
                temp = [float(seg) for seg in temp if seg != '']
                if len(temp) == 2:
                    currentPosition, storedPosition = temp, temp
                    #output.append(("<path d=\"M" + str(currentPosition[0]) + " " + str(currentPosition[1]) + "\"/>"))
                else:
                    # Error in M command format
                    pass
            
            elif line[0] == "m":
                temp = line.split(" ")
                temp[0] =float(temp[0][1:])
                temp = [float(seg) for seg in temp if seg != '']
                if len(temp) == 2:
                    instruct, currentPosition = operatem(temp, currentPosition)
                    storedPosition = currentPosition
                    #output.append(("<path d=\"M" + instruct + "\"/>"))
                else:
                    # Error in m command format
                    pass


            elif line[0] == "c":
                temp = line.split(" ")
                temp[0] = temp[0][1:]
                temp = [float(seg) for seg in temp if seg != '']
                
                if len(temp) % 6 == 0:
                    for k in range(0, int(len(temp)), 6):
                        temp2 = ("<path d=\"M " + str(currentPosition[0]) + " " + str(currentPosition[1]) + " C")
                        arc, currentPosition = operatec(temp[k:k+6], currentPosition)
                        output.append((temp2 + arc + "\"/>"))
                else:
                    # Error in c command format
                    pass
            
            elif line[0] == "l":
                temp = line.split(" ")
                temp[0] = temp[0][1:]
                temp = [float(seg) for seg in temp if seg != '']
                if len(temp) % 2 == 0:
                    for k in range(0, int(len(temp)), 2):
                        temp2=("<path d=\"M " + str(currentPosition[0]) + " " + str(currentPosition[1]) + " L")
                        lineOutput, currentPosition = operatel(temp[k:k+2], currentPosition)
                        output.append((temp2 + lineOutput + "\"/>"))
                else:
                    # Error in l command format
                    pass

            elif line[0] in ("Z", "z"):
                temp2 = ("<path d=\"L " + str(currentPosition[0]) + " " + str(currentPosition[1]) + " " + str(storedPosition[0]) + " " + str(storedPosition[1]) + "\"/>")
                output.append(temp2)
                currentPosition = storedPosition

            else:
                # Unrecognized command
                pass

    output.append("\n</g>\n</svg>")
    return output

def writeSvg(pathList, fileName):
    with open(fileName, "w") as file:
        for line in pathList:
            file.write(line)
            file.write("\n")