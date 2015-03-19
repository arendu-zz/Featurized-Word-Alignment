import sys

if __name__ == "__main__":
    inputdir = sys.argv[1]
    outputdir = sys.argv[1] + ".out"

    inputfile = open(inputdir)
    outputfile = open(outputdir, 'w')

    for rline in inputfile:
        rline = rline.replace(" ", "\t")
        wline = "EMISSION\t" + rline
        outputfile.write(wline)
    inputfile.close()
    outputfile.close()
