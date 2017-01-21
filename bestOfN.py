
import sys
import AF as AF
from tqdm import tqdm


def main(N=5):
	
	#if len(sys.argv) > 1:
	#    N = int(sys.argv[1])
	#elif len(argv)>0:
	#    N = int(argv[0])
	#else:
	

	x={}
	bestObj = float("inf")
	bestPtr = 0
	for i in tqdm(range(0,N)):
		x[i] = AF.af(1,[0,0,0])
		if x[i].obj < bestObj and x[i].obj>0:
		    bestPtr = i
		    bestObj = x[i].obj
	print x[bestPtr]


if __name__ == "__main__":
   main(sys.argv[1:])
