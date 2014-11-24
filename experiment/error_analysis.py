import argparse
import sys
import pdb

parser = argparse.ArgumentParser(description="Analyze the errors made in word alignment")
parser.add_argument("--sys1", help="the output for regular model (column format, index)", required=True)
parser.add_argument("--sys2", help="the output for featurized model (column format, index)", required=True)
parser.add_argument("--gold", help="the gold standard for evaluation (column format, index)", required=True)
parser.add_argument("--src", help="the source file of the alignment", required=True)
parser.add_argument("--tar", help="the target file of the alignment", required=True)
parser.add_argument("--prob1", help="the probability file for the regular model", required=True)
parser.add_argument("--prob2", help="the probability file for the featurized model", required=True)
parser.add_argument("--out", help="the analysis output")

args = parser.parse_args()

if __name__ == "__main__":
	# read in probability table
	probfile1 = open(args.prob1)
	probfile2 = open(args.prob2)
	probchart1 = {}
	probchart2 = {}
	# read the probability files
	for line in probfile1:
		if line.strip()[0] == '#':
			continue
		tokens = line.strip().split('\t')
		problist = probchart1.get(tokens[1], [])
		problist.append((tokens[2], float(tokens[3])))
		probchart1[tokens[1]] = problist
	for line in probfile2:
		if line.strip()[0] == '#':
			continue
		tokens = line.strip().split('\t')
		problist = probchart2.get(tokens[1], [])
		problist.append((tokens[2], float(tokens[3])))
		probchart2[tokens[1]] = problist

	# the index of the sentence we are processing
	sentN = 1

	# the sum of difference of rank and probability between systems
	# for the calculation of average
	rank_difference_sum = 0
	prob_difference_sum = 0.0

	# "valid error" means errors which are not caused by randomness
	valid_errorN = 0

	sys1file = open(args.sys1)
	sys2file = open(args.sys2)
	goldfile = open(args.gold)
	srcfile = open(args.src)
	tarfile = open(args.tar)
	if args.out == None:
		outfile = sys.stdout
	else:
		outfile = open(args.out)
	outfile.write("foreign\t\twrong\t\tcorrect\t\twrank\tcrank\trankdiff\tvocabsize\twprob\t\tcprob\t\tprobdiff\n")
	sys1line = sys1file.readline().strip()
	sys2line = sys2file.readline().strip()
	goldline = goldfile.readline().strip()
	# each iteration process one sentence
	while goldline != "":
		srcline = srcfile.readline().strip()
		tarline = tarfile.readline().strip()
		if srcline == "" or tarline == "":
			sys.stderr.write("source/target file line numbers must agree with alignment outputs!\n")
			sys.exit(1)
		sys1indexes = sys1line.split(' ')
		sys2indexes = sys2line.split(' ')
		goldindexes = goldline.split(' ')
		srctokens = srcline.split(' ')
		tartokens = tarline.split(' ')

		# read the gold alignments for this sentence
		golds = {}
		while not goldline == "" and int(goldindexes[0]) == sentN:
			golds[int(goldindexes[2])] = int(goldindexes[1])
			goldline = goldfile.readline().strip()
			goldindexes = goldline.split(' ')

		# read the system1 output for this sentence
		sys1 = {}
		while not sys1line == "" and int(sys1indexes[0]) == sentN:
			sys1[int(sys1indexes[2])] = int(sys1indexes[1])
			sys1line = sys1file.readline().strip()
			sys1indexes = sys1line.split(' ')

		# read the system2 output for this sentence
		sys2 = {}
		while not sys2line == "" and int(sys2indexes[0]) == sentN:
			sys2[int(sys2indexes[2])] = int(sys2indexes[1])
			sys2line = sys2file.readline().strip()
			sys2indexes = sys2line.split(' ')

		# collect the errors (not necessarily valid!)
		errors = []
		for tarindex in range(1, len(srctokens) + 1):
			# 0 stands for NULL
			srcgold = golds.get(tarindex, 0)
			srcsys1 = sys1.get(tarindex, 0)
			srcsys2 = sys2.get(tarindex, 0)
			if srcsys1 == srcgold and not srcsys2 == srcgold:
				errors.append((tarindex, srcsys1, srcsys2))

		for error in errors:
			# get the tokens
			if error[1] > 0:
				sys1srctoken = srctokens[error[1] - 1]
			else:
				sys1srctoken = "NULL"
			if error[2] > 0:
				sys2srctoken = srctokens[error[2] - 1]
			else:
				sys2srctoken = "NULL"
			tartoken = tartokens[error[0] - 1]

			# get probability list for certain target token
			problist1 = sorted(probchart1.get(tartoken), key=lambda prob: prob[1], reverse=True)
			problist2 = sorted(probchart2.get(tartoken), key=lambda prob: prob[1], reverse=True)

			# get the prob and the rank for the output
			wrong_rank = 0
			wrong_prob = 0.0
			correct_rank = 0
			correct_prob = 0.0
			tuple_index = 0
			for (srctoken, prob) in problist1:
				if srctoken == sys1srctoken:
					correct_prob = prob
				tuple_index += 1
			tuple_index = 0
			for (srctoken, prob) in problist2:
				if srctoken == sys2srctoken:
					wrong_rank = tuple_index
					wrong_prob = prob
				if srctoken == sys1srctoken:
					correct_rank = tuple_index
					# this is probability for the system1
					# output in the probability file of
					# system2
					correct_prob2 = prob
				tuple_index += 1
			# if the error is not valid, throw away
			if correct_prob2 - wrong_prob == 0:
				continue
			valid_errorN += 1
			rank_difference_sum += (correct_rank - wrong_rank)
			prob_difference_sum += (correct_prob - wrong_prob)

			# output error analysis information
			if len(str(error[0])) + len(tartoken) < 7:
				line = str(error[0]) + "/" + tartoken + "\t\t"
			elif len(str(error[0])) + len(tartoken) >=7 and len(str(error[0])) + len(tartoken) <= 14:
				line = str(error[0]) + "/" + tartoken + "\t"
			else:
				line = str(error[0]) + "/" + tartoken[0:12] + "\t"

			if len(str(error[0])) + len(sys2srctoken) < 7:
				line += (str(error[0]) + "/" + sys2srctoken + "\t\t")
			elif len(str(error[0])) + len(sys2srctoken) >=7 and len(str(error[0])) + len(sys2srctoken) <= 14:
				line += (str(error[0]) + "/" + sys2srctoken + "\t")
			else:
				line += (str(error[0]) + "/" + sys2srctoken[0:12] + "\t")

			if len(str(error[0])) + len(sys1srctoken) < 7:
				line += (str(error[0]) + "/" + sys1srctoken + "\t\t")
			elif len(str(error[0])) + len(sys1srctoken) >=7 and len(str(error[0])) + len(sys1srctoken) <= 14:
				line += (str(error[0]) + "/" + sys1srctoken + "\t")
			else:
				line += (str(error[0]) + "/" + sys1srctoken[0:12] + "\t")

			line += (str(wrong_rank) + "\t")
			line += (str(correct_rank) + "\t")
			line += (str(correct_rank - wrong_rank) + "\t\t")
			line += (str(len(problist2)) + "\t\t")
			line += (str(wrong_prob) + "\t")
			if len(str(wrong_prob)) < 8:
				line += "\t"
			line += (str(correct_prob) + "\t")
			if len(str(correct_prob)) < 8:
				line += "\t"
			line += (str(wrong_prob - correct_prob) + "\t")
			line += "\n"
			outfile.write(line)
		sentN += 1
	line = "average rank difference: " + str(float(rank_difference_sum) / valid_errorN) + "\n"
	outfile.write(line)
	line = "average prob difference: " + str(float(prob_difference_sum) / valid_errorN) + "\n"
	outfile.write(line)
	sys1file.close()
	sys2file.close()
	goldfile.close()
	srcfile.close()
	tarfile.close()
	outfile.close()
