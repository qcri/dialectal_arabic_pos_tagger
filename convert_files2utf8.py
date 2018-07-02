#!/usr/bin/env python
# -*- coding: latin-1 -*-

#import unicodedata
import codecs
import sys
import optparse

_unicode = u"\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649\u064b\u064d\u064f\u0651\u0671"
_safebuckwalter = u"MWQbtjGxVzcDZg_qlnwyNaio`PJCOIVApvHdrsSTEfkmhYFKu~{"

def toSafeBuckWalter(s):
    return s.translate(_safeforwardMap)

def fromSafeBuckWalter(s):
    return s.translate(_safebackwardMap)

_safeforwardMap = {ord(a):b for a,b in zip(_unicode, _safebuckwalter)}
_safebackwardMap = {ord(b):a for a,b in zip(_unicode, _safebuckwalter)}


def main():
	print("Processing "+sys.argv[1])
	fOut = open(sys.argv[1] + '.utf8', 'w')
	sep ='\t'
	with open(sys.argv[1], 'r') as fIn:
		if(sys.argv[1].endswith('segmented-vectors')):
			sep=' '
		for line in fIn:
			input = line.strip().split(sep)[0]
			if(input not in ('TB','EOS', 'UNKNOWN', 'PADDING') and not input.startswith('http') and not input.startswith('@') and not input.startswith('#')):
				input = fromSafeBuckWalter(input)
			fOut.write(input+sep+' '.join(line.strip().split(sep)[1:])+'\n')
	fOut.close()
	fIn.close()





if __name__ == "__main__":
	main()