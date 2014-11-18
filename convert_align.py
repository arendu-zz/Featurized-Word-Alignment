import sys

if __name__ == "__main__":
	if len(sys.argv) > 3 or len(sys.argv) < 2:
		sys.stdout.write("usage: convert_align input [output]")
	elif len(sys.argv) == 2:
		inputdir = sys.argv[1]
		outputdir = inputdir + ".out"
	else:
		inputdir = sys.argv[1]
		outputdir = sys.argv[2]

	inputfile = open(inputdir)
	outputfile = open(outputdir, 'w')
	sentN = 1
	for line in inputfile:
		line = line.strip()
		if line.strip()[0:1] == "#":
			continue
		aligns = line.split(" ")
		for align in aligns:
			indexes = align.split("-")
			newalign = str(sentN) + " " + str(int(indexes[0]) + 1) + " " + str(int(indexes[1]) + 1) + "\n"
			outputfile.write(newalign)
		sentN += 1
	inputfile.close()
	outputfile.close()
