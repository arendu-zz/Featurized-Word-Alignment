import argparse
import sys

parser = argparse.ArgumentParser(description="Evaluate the performance of unsupervised POS-tagger")
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#	help='an integer for the accumulator')
parser.add_argument("--sys", "-s", help="the tagger output", required=True)
parser.add_argument("--gold", "-g", help="the gold standard for evaluation", required=True)
parser.add_argument("--assignment", "-a", default=False, action="store_true", help="print the assignment between the output cluster and the standard label")

args = parser.parse_args()
# print args.sys, args.gold, args.assignment

# collect all the labels in the gold standard and their frequencies
tags = {}
fgold = open(args.gold, 'r')
for line in fgold:
	tokens = line.strip().split(' ')
	for token in tokens:
		if token in tags:
			tags[token] += 1
		else:
			tags[token] = 1
fgold.close()

# collect all the clusters in the system output and their frequencies and their joint occurence with labels
lineNumber = 0
clusters = {}
occurence = {}
fsys = open(args.sys, 'r')
fgold = open(args.gold, 'r')

for (sline, gline) in zip(fsys, fgold):
	stokens = sline.rstrip().split(' ')
	gtokens = gline.rstrip().split(' ')
	if (sline == None and gline == None):
		sys.stderr.write("on line " + str(lineNumber) + ": system output and gold standard has different line numbers.\n")
		exit(1)

	if (len(stokens) != len(gtokens)):
		sys.stderr.write("on line " + str(lineNumber) + ": system output and gold standard has different number of tokens. I'll skip this one.\n")
		continue
	else:
		for (stoken, gtoken) in zip(stokens, gtokens):
			# collect cluster
			if stoken in clusters:
				clusters[stoken] += 1
			else:
				clusters[stoken] = 1

			# collect occurence
			if (stoken, gtoken) in occurence:
				occurence[(stoken, gtoken)] += 1
			else:
				occurence[(stoken, gtoken)] = 1
	lineNumber += 1

# transform the joint occurence into precentage
for (stoken, gtoken) in occurence:
	occurence[(stoken, gtoken)] /= float(clusters[stoken])

# build correspondencies
assignment = {}
unassignedTags = tags.keys()
for cluster in clusters:
	# There are unassigned tags
	if unassignedTags:
		score = {}
		for tag in unassignedTags:
			score[(cluster, tag)] = occurence.get((cluster, tag), 0.0)
		sortedScore = sorted(score.items(), lambda x, y: cmp(x[1], y[1]), reverse = True)
		for item in sortedScore:
			assignment[item[0][0]] = item[0][1]
			unassignedTags.remove(item[0][1])
			break

# evaluate
correct = 0.0
total = sum(tags.values())
for item in assignment.items():
	if args.assignment:
		sys.stdout.write(str(item) + "\n")
	correct += (occurence.get(item, 0.0) * clusters[item[0]])
sys.stdout.write("precision = " + str(correct / total) + "\n")
