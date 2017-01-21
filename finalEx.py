from assignmentFunction import *
import timeit
from tqdm import tqdm
from itertools import product
from scipy.stats import describe
import AF as AF

rs = range(0,6) #restrictions
dr = range(0,6) #directeds
rj = range(0,6) #rejects
it = range(0,10)#iterations
me = range(1,4) #method

KD_Off= range(74,160)
B_Off = range(0, 74)
Officers = list(set().union(KD_Off,B_Off))

f = open('results.csv', 'a')
g = open('matchings.csv', 'a')

f.write("method, obj, restrictions,directeds,rejects,iter,changes, time,feasible,ranksnobs,ranksmean,ranksmin,ranksmax,ranksvar,control\n")
for o in Officers:
	g.write(str(o)+',')
g.write('feasible, control')

num = 0

for i,j,k,l,m in tqdm(product(rs,dr,rj,it,me)):
    num += 1
    start_time = timeit.default_timer()
    s = AF.af(m,[i,j,k])
    t = timeit.default_timer() - start_time
    st = describe(AF.getRanks(s.finalSolution))

    f.write(str(m)+ ','+ str(s.obj)+ ',' + str(i)+','+str(j)+','+str(k)+','+str(l)+','+str(s.changes)+','+str(t)+','+str(s.feasible) +','+str(st.nobs)+','+str(st.mean)+','+str(st.minmax[0])+','+str(st.minmax[1])+','+str(st.variance)+','+ str(num)+'\n')

    for o in Officers:
	g.write(str(s.finalSolution[o])+',')
    g.write(str(s.feasible)+','+str(num)+'\n')

f.close()
g.close()
print 'Complete!                                                                                \n'
 
