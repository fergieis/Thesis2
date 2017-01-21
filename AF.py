#from __future__ import division
import numpy as np
import pandas as pd
import pickle as pkl


#two helper functions to save and load python objects
def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pkl.load(f)

#Solution Class provides means of return more than one value
class Solution:
    def __init__(self, fS, rSF, c): #add msg?
        self.finalSolution = fS
	#self.computationTime = cT
	self.resultStatusFlag = rSF
	self.changes= c
	self.obj = 0.0
	#self.resultStatusMsg = ...
	self.feasible = 1

    def __str__(self): #when called via print
	return str([self.finalSolution, self.obj, self.feasible, self.changes])
	
    def __repr__(self): #when called interactively
	return str([self.finalSolution, self.obj, self.feasible, self.changes])


def setup():
    LPGetYearGroup()
    SMAGetPrefs()
    LPGetC()

#Generates 'p.pkl' file containing Officer preference dictionary
#Only needs to be run once on a machine
#setup for use as a stack (pop/append) in reverse preference order.
#Stack contains sublists for each level of indifference (weak preference order)
def SMAGetPrefs():
    import numpy as np
    import pandas as pd

    p_kd = {}
    p_b = {}
    KD_Off= range(74,160)
    B_Off = range(0, 74)
    KD_Ass= range(0,71)
    B_Ass = range(73,139)
    D_Ass = range(139, 162)
    KD_D_Ass = B_Ass[0:15]

    Officers = list(set().union(KD_Off,B_Off))
    Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))

    raw_pref = pd.read_csv('RawPreferences2.csv', dtype='str')
    raw_pref.fillna(max(KD_Ass)+1,inplace=True) #worst case, 72nd choice
    for o in Officers:
	p_kd[o] = map(int,raw_pref.iloc[o,KD_Ass])
	for i in range(0,2):
	    p_kd[o].append(999) #missing assignments 71/72
	for a in KD_D_Ass:	
	    p_kd[o].append(a)
	p_b[o] = map(int,raw_pref.iloc[o,B_Ass])
	for a in D_Ass:
            p_b[o].append(a)

    save_obj(p_kd, 'p_kd')
    save_obj(p_b, 'p_b')

    Officers = list(set().union(KD_Off,B_Off))
    Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))

    r_kd = {}
    r_b = {}

    raw_pref = pd.read_csv('RawPreferences2.csv', dtype='str')
    for o in Officers:
		
	r=raw_pref.iloc[o,KD_Ass]
	r.fillna(max(KD_Ass)+1,inplace=True) #worst case scenario
	r_kd[o] = map(int,r)

	
        r = raw_pref.iloc[o,B_Ass]
	r.fillna(max(B_Ass)+1,inplace=True)
	r_b[o] = map(int,r)
	for a in D_Ass:
            r_b[o].append(0)
    save_obj(r_kd, 'r_kd')
    save_obj(r_b, 'r_b')


#Generates 'y.pkl' file containing yeargroup list. 
#Only needs to be run once on a machine
def LPGetYearGroup():
    d = pd.read_csv('OfficerData.csv')
    y = d['YG']
    y = y.values.tolist()
    save_obj(y, 'y')

    yg={}
    for o, year in enumerate(y):
	try:
	    yg[year].append(o)
	except (KeyError, NameError):
	    yg[year]=[o]
    save_obj(yg,'yg')

#Generates 'C.pkl' file containing cost coefficient list. 
#Only needs to be run once on a machine
def LPGetC():
    KD_Off= range(74,160)
    B_Off = range(0, 74)
    KD_Ass= range(0,71)
    B_Ass = range(73,139)
    D_Ass = range(139, 162)

    y = load_obj('y')
    years = {}
    for i,year in enumerate(y):
	years[i] = (y[i]-min(y))

    Officers = list(set().union(KD_Off,B_Off))
    Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))
    C = {}
    prefs ={} 
    officer_preference = pd.read_csv('OfficerPrefs.csv')	
    for o in Officers: 
        prefs[o] = officer_preference[str(o)]
        for a in Assignments:
            if a in list(D_Ass):
		C[(o,a)] = 1.6 * (max(years)-years[o])
	    elif not np.isnan(prefs[o][a]):		
		if o in KD_Off and a in KD_Ass:
		    penalty = 160*years[o]
		elif o in KD_Off and a in B_Ass:
		    penalty = 160 *(max(years)-years[o])
		elif o in B_Off and a in KD_Ass:
		    penalty = 160
	        elif o in B_Off and a in B_Ass:		
		    penalty = 0
		else:
		    pass
		C[(o,a)] = prefs[o][a] + penalty
	    else: #mildly infeasible, no KD prefs avail
		C[(o,a)]= 999
    save_obj(C,'C')


#SMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMA
# SMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMASMA  SMA  SMA

def SMA(allowableChanges):
    import random as rnd
    sol = Solution([0,0], 1, 0)	
    #try:
    if 1:
        #sets
        KD_Off= range(74,160)
        KD_Ass= range(0,71)
	B_Off = range(0, 74)
	B_Ass = range(73,139)
	D_Ass = range(139, 162)	
	Officers = list(set().union(KD_Off,B_Off))
	#Assignments = list(set().union(KD_Ass, B_Ass))
 	Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))

	#Match KD Assignments
	KD_D_Ass = B_Ass[0:15]
	p_kd = load_obj('p_kd')
	p_b = load_obj('p_b')
	yg = load_obj('yg')
	y = load_obj('y')
	smaA = load_obj('smaA')
	KD_OtoA = {}
	BOtoA = {}
	res_list = {}

	#allowableChanges[0]: Assignment Restrictions	
	availO = set(Officers)	
	restrictions = rnd.sample(list(availO), allowableChanges[0])
	for restriction in restrictions:
	    res_list[restriction] = rnd.sample(Assignments, int(rnd.uniform(.05,.1) * len(Assignments)))
	    for ass in res_list[restriction]:
		if ass in KD_Ass and restriction in KD_Off:
		    p_kd[restriction][ass]  = max(Assignments) + 2
		elif ass in B_Ass:
		    p_b[restriction][ass-min(B_Ass)] = max(Assignments) + 2
		else:
                    pass	        



	#allowableChanges[1]: Directed Assignment
	taken = []
	x = {}
	availO -= set(restrictions)
	availA = set().union(KD_Ass, B_Ass, D_Ass)
        directeds = rnd.sample(list(availO), allowableChanges[1])
	for directed in directeds:
	    #if directed in KD_Off:
		#KD_Off.remove(directed)
	    #else:
		#B_Off.remove(directed)
	    #x[directed]=rnd.sample(list(availA), 1)[0]
	    if directed in KD_Off:
		x[directed]=rnd.sample(list(set(availA)-set(B_Ass)), 1)[0]
		KD_OtoA[directed] = x[directed]	
	    else:
		x[directed]=rnd.sample(list(set(availA)-set(KD_Ass)), 1)[0]
            	BOtoA[directed] = x[directed]
	    taken += [x[directed]]
	    availA -= set(taken)


	#allowableChanges[2]: Rejected Match
	availO -= set(directeds)
	rejects = []
	while len(rejects) < allowableChanges[2]:
	    reject = rnd.sample(list(availO), 1)[0]
	    if int(smaA[reject]) != 999:
		rejects.append(reject)
		availO.remove(reject)
		if reject in KD_Off and smaA[reject] in KD_Ass:
		    p_kd[reject][smaA[reject]] = max(Assignments)+2
		else: #if reject in KD_Off and smaA[reject] in B_Ass:
		    #p_b[reject].remove(smaA[reject])
		    p_b[reject][smaA[reject]-min(B_Ass)] = max(Assignments) + 2

			
		
	del availO
	del availA

	#Starting KD matching 
	opref = {}
	apref = {}
	unmatched = list(set(KD_Off)-set(directeds)) #listing of unmatched officers
	noKD = [] #bookkeeping for KD Officers not getting a KD Assignment
	#kdNotTaken = list(set().union(set(KD_Ass),set(KD_D_Ass) )- set(taken))
	#for o in KD_Off:
	level= {}

	for assignment in KD_Ass:
	    apref[assignment] = {}
	    preference = 0
	    for year in sorted(yg.keys()):
		preference += 1
		for officer in yg[year]:
		    apref[assignment][officer] = preference
	
		
	for assignment in KD_D_Ass:
	    apref[assignment] = {}
	    for officer in KD_Off:   
	 	apref[assignment][officer] = -y[officer]

	for officer in KD_Off:
	    opref[officer] = p_kd[officer]
	    level[officer] = 0
	    if officer not in directeds:
		KD_OtoA[officer] = -1

	
#	for o in directeds:
#	    if x[o] in KD_Ass:
#		apref[x[o]][o] = -9999 #Will never get "bumped"
#		KD_OtoA[o] = x[o]
	
	while unmatched:
	    officer = rnd.choice(unmatched)
	    if level[officer] > 100: #hack that fixes overlap 
		noKD.append(officer) #of KD_D_Ass and B_Ass
		break		     #only occaisionally needed
	    try:
	        possibility = opref[officer].index(level[officer])
	 	opref[officer][possibility]=-1
	    except ValueError: #level not found in opref.index()
		level[officer] +=1
		continue

	    if officer in directeds or possibility in list(taken):
		continue

	    try:
		incumbent = KD_OtoA.keys()[KD_OtoA.values().index(possibility)]
	    except ValueError:
	        incumbent = -1	
	
	    if officer not in rejects or possibility != smaA[officer]:
		possibilitypref = apref[possibility]
	    else: #don't assign a reject to his incumbent match.
		continue
		



	    #no incumbent	
	    if incumbent == -1:
	  	unmatched.remove(officer)			
		KD_OtoA[officer] = possibility
		if possibility in B_Ass:
		    noKD.append(officer)
				
	    #if assignment prefers officer to incumbent	
	    elif possibilitypref[officer] < possibilitypref[incumbent]:
		KD_OtoA[incumbent] = -1
		if possibility in B_Ass:
		    noKD.remove(incumbent)
		    noKD.append(officer)
		unmatched.append(incumbent)
		KD_OtoA[officer] = possibility
		unmatched.remove(officer)	
	    else: #incumbent is preferred, do nothing
		pass

	
#-----------------------------------------
	del p_kd	#other garbage collect
	del yg
#-----------------------------------------

	#Setup for Broad. Matching
	opref = {}
	apref = {}
	level = {}
	B_OtoA ={}

	p_b = load_obj('p_b')
	unmatched = list(set(noKD + B_Off)-set(directeds))
	for o in unmatched:
	    opref[o] = p_b[o]
	    if officer not in directeds:
	        B_OtoA[o] = -1
	        level[o] = 0
	
	for assignment in B_Ass:
	    apref[assignment]={}
	    for officer in unmatched:
		if officer in B_Off:
		    apref[assignment][officer] = 1
		else:
		   apref[assignment][officer] = 2   
	
   	for assignment in D_Ass:
	    apref[assignment] = {}
	    for officer in unmatched:
		apref[assignment][officer] = -y[officer]
	
#	for o in directeds:
#	    if x[o] in B_Off:
#		apref[x[o]]= apref[160]
#		apref[x[o]][o] = -9999
	
	while unmatched:
	    officer = rnd.choice(unmatched)
	    #print str(officer)+"\t"+str(level[officer])+"\t"+str(opref[officer])
	    #officer = unmatched.pop()	
	    if level[officer] > max(Assignments)+2:
		print B_OtoA		
		print str(level[officer]) +">"+str(max(Assignments)+2)
		raise ValueError
	    try:
	        possibility = opref[officer].index(level[officer])+min(B_Ass)
	 	opref[officer][possibility-min(B_Ass)]=-1
	    except ValueError: #level not found in opref.index()
		level[officer] +=1
		continue
	    try:
		incumbent = B_OtoA.keys()[B_OtoA.values().index(possibility)]
	    except ValueError:
	        incumbent = -1	
	
	    if officer not in rejects or possibility != smaA[officer]:
		possibilitypref = apref[possibility]
	    else: #don't assign a reject to his incumbent match.
		continue
		
	    if officer in directeds or possibility in list(taken):
		continue

	    #no incumbent	
	    if incumbent == -1:
		unmatched.remove(officer)			
		B_OtoA[officer] = possibility
		if possibility in KD_D_Ass:
		    noKD.append(officer)
				
	    #if assignment prefers officer to incumbent	
	    elif possibilitypref[officer] < possibilitypref[incumbent]:
		B_OtoA[incumbent] = -1
		if possibility in KD_D_Ass:
		    noKD.remove(incumbent)
		    noKD.append(officer)
		unmatched.append(incumbent)
		B_OtoA[officer] = possibility
		unmatched.remove(officer)	
	    else: #incumbent is preferred, do nothing
		    pass

	    

#+++++++++++++++++++++++++++++++++++++++++++++++++
	for o in KD_Off:
	    if o in noKD and o not in directeds:
		
		x[o] = B_OtoA[o]
	    elif o not in directeds:
		x[o] = KD_OtoA[o]
	for o in B_Off:
	    if o not in directeds:
	        x[o] = B_OtoA[o]
	    #print "O:"+str(o)+" A:"+str(KD_OtoA[o])
	sol.finalSolution = []
	Officers = list(set().union(KD_Off,B_Off))
	for i in Officers:
	    if x[i] in D_Ass or x[i] == -1:
	   	sol.finalSolution.append(int(999))
	    else:	
	        sol.finalSolution.append(x[i])
	for restriction in restrictions:
	    if x[restriction] in res_list[restriction]:
	        sol.feasible = 0

	for reject in rejects:
	    if x[reject] == smaA[reject]:
		sol.feasible = 0


	sol.resultStatusFlag = 0
	#save_obj(sol.finalSolution, 'smaA') #saved first time, used later
		
	sol.changes = [sum(i != j for i, j in zip(sol.finalSolution, smaA))][0]

    #except IndexError:
    #	print "Index Error"        
    #	sol = Solution(-1* np.ones(160), -1, -1)	
	
    return sol

#AFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAF
#AFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAFAF

def af(methodFlag=1, allowableChanges=[0,0,0]): #don't need init solution
    #from math import floor, sqrt
    #for non-integer division, convert to floats!	
    #import pandas as pd
    import numpy as np
    import gurobipy as gp
    import random as rnd
    
    
    sol = Solution([0,0], 1, 0)	


    #Because Python doesn't have select-case
    if methodFlag == 0:   #SMA"Warmstart" -- may drop
	print "not implemented"
    

    elif methodFlag == 1: #SMAColdstart
	sol = bestOfN(5, allowableChanges)

    elif methodFlag == 2: #LPWarmstart
	kd_m = gp.Model()	
	y = load_obj('y')
	lpA = load_obj('lpA')

        #sets
        KD_Off= range(74,160)
        B_Off = range(0, 74)
        KD_Ass= range(0,71)
        B_Ass = range(73,139)
        D_Ass = range(139, 162)
	D_KD_Ass = range(139,154)
	
	Officers = list(set().union(KD_Off,B_Off))
	KD_Assignments = list(set().union(KD_Ass,D_KD_Ass))	
	Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))
	
        #handlemods/random perturbations

	#allowableChanges[0]: Assignment Restrictions	
	restrictions = rnd.sample(Officers, allowableChanges[0])
	A_r = {}
	for restriction in restrictions:
	    A_r[restriction] = rnd.sample(Assignments, int(rnd.uniform(.05,.1) * len(Assignments)))

	#allowableChanges[1]: Directed Assignment
	A_d = {}
	availO = set(Officers)
	availO -= set(restrictions)
        directeds = rnd.sample(list(availO), allowableChanges[1])
	availA = set(Assignments)
	for directed in directeds:
	    availA -= set(A_d.values())
	    A_d[directed] = rnd.sample(list(availA), 1)
	    A_d[directed]=A_d[directed][0]

	#allowableChanges[2]: Rejected Match
	availO -= set(directeds)
	rejects = []
	while len(rejects) < allowableChanges[2]:
	    reject = rnd.sample(list(availO), 1)[0]
	    if int(lpA[reject]) != 999:
		rejects.append(reject)
		availO.remove(reject)

	#allowableChanges[3]: Unforecast Assignment
	#Need to remove all references, these are just the dummy assignments
	#mildly embarassing... but reference in write up

	
	#Phase I, slot KD Offs/objective  f_1
	x={}
	for o in KD_Off:
	    for a in KD_Assignments:
		x[(o,a)] = kd_m.addVar(vtype=gp.GRB.BINARY, 
			   obj = y[o],
                           name="O{0:03d}".format(o) + 
			   "A{0:03d}".format(a))
		#if lpA[o] == a:
		#	x[(o,a)].Start = 1
		#else:
		#	x[(o,a)].Start = 0

	kd_m.ModelSense = gp.GRB.MINIMIZE #MINIMIZE

        for a in KD_Assignments:
            kd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in KD_Off),gp.GRB.EQUAL,1)	
	for o in KD_Off:
	    if o in list(restrictions):
		if not list(set(A_r[o])&set(KD_Assignments)):
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in list(set(A_r[o])&set(KD_Ass))), gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)  
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(directeds):
		if A_d[o] in list(KD_Ass):
	            kd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(rejects):
		if int(lpA[o]) in list(KD_Ass):
		    kd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
	    else:
	        kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)

	kd_m.update()
	kd_m.setParam('OutputFlag', False)
	#kd_m.write('lp.mps')
	kd_m.optimize()
	try:
	    y_star = kd_m.objVal
	except:
	    #print "Error in y_star"    
	    return Solution(-1* np.ones(160), -1, -1)
	#Continue with phase2, slot everyone- obj f_2

	del kd_m

        C = load_obj('C')
	x = {} #reallocating memory
	bd_m = gp.Model()

        for o in Officers: 
	    for a in Assignments:
		x[(o,a)] = bd_m.addVar(vtype=gp.GRB.BINARY, obj=C[(o,a)],name="O{0:03d}".format(o) + "A{0:03d}".format(a))
		
        bd_m.ModelSense = gp.GRB.MINIMIZE #-1
	for a in Assignments:
            bd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in Officers),gp.GRB.EQUAL,1)	
	for o in Officers:
	    if o in list(restrictions):
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in A_r[o]), gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(directeds):
	        bd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(rejects):
		bd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    else:
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	
#	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, 1.0015* y_star) 
	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, y_star)
	bd_m.setParam('OutputFlag', False)
	bd_m.update()
	bd_m.optimize()
	#print bd_m.Status
	#unassigned =[]
	sol.finalSolution = np.zeros(160)
        #try:
	for v in bd_m.getVars():
	        if v.x >0:
	            if int(v.varName[-3:])>=139 or int(v.varName[-3:]) in [71,72]:
	                #unassigned.append(int(v.varName[1:4]))
		        sol.finalSolution[int(v.varName[1:4])] = 999
		    else:		
		        sol.finalSolution[int(v.varName[1:4])] = int(v.varName[-3:])
	sol.finalSolution = list(sol.finalSolution.astype(int))
	sol.resultStatusFlag = 0 #Good Execution
	#save_obj(sol.finalSolution, 'lpA') #saved locally first time and referenced later
	    #lpA = load_obj('lpA')
	sol.changes = sum(i != j for i, j in zip(sol.finalSolution, lpA))
	sol.obj = bd_m.objVal

        #except:
	#    sol = Solution(-1* np.ones(160), -1, -1) 





    elif methodFlag == 3: #LPColdstart
	kd_m = gp.Model()	
	y = load_obj('y')
	lpA = load_obj('lpA')

        #sets
        KD_Off= range(74,160)
        B_Off = range(0, 74)
        KD_Ass= range(0,71)
        B_Ass = range(73,139)
        D_Ass = range(139, 162)
	D_KD_Ass = range(139,154)
	Officers = list(set().union(KD_Off,B_Off))
	KD_Assignments = list(set().union(KD_Ass,D_KD_Ass))	
	Assignments = list(set().union(KD_Ass, B_Ass, D_Ass))
	
        #handlemods

	#allowableChanges[0]: Assignment Restrictions	
	restrictions = rnd.sample(Officers, allowableChanges[0])
	A_r = {}
	for restriction in restrictions:
	    A_r[restriction] = rnd.sample(Assignments, int(rnd.uniform(.05,.1) * len(Assignments)))

	#allowableChanges[1]: Directed Assignment
	A_d = {}
	availO = set(Officers)
	availO -= set(restrictions)
        directeds = rnd.sample(list(availO), allowableChanges[1])
	availA = set(Assignments)
	for directed in directeds:
	    availA -= set(A_d.values())
	    A_d[directed] = rnd.sample(list(availA), 1)
	    A_d[directed]=A_d[directed][0]

	#allowableChanges[2]: Rejected Match
	availO -= set(directeds)
	rejects = []
	while len(rejects) < allowableChanges[2]:
	    reject = rnd.sample(list(availO), 1)[0]
	    if int(lpA[reject]) != 999:
		rejects.append(reject)
		availO.remove(reject)

	#allowableChanges[3]: Unforecast Assignment
	#Need to remove all references, these are just the dummy assignments
	#mildly embarassing... but reference in write up

	
	#Phase I, slot KD Offs/objective  f_1
	x={}
	for o in KD_Off:
	    for a in KD_Assignments:
		x[(o,a)] = kd_m.addVar(vtype=gp.GRB.BINARY, 
			   obj = y[o],
                           name="O{0:03d}".format(o) + 
			   "A{0:03d}".format(a))
	kd_m.ModelSense = gp.GRB.MINIMIZE #MINIMIZE

        for a in KD_Assignments:
            kd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in KD_Off),gp.GRB.EQUAL,1)	
	for o in KD_Off:
	    if o in list(restrictions):
		if not list(set(A_r[o])&set(KD_Assignments)):
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in list(set(A_r[o])&set(KD_Ass))), gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)  
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(directeds):
		if A_d[o] in list(KD_Assignments):
	            kd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,0)
	    elif o in list(rejects):
		if int(lpA[o]) in list(KD_Ass):
		    kd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
		else:
		    kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)
	    else:
	        kd_m.addConstr(gp.quicksum(x[(o,a)] for a in KD_Assignments),gp.GRB.EQUAL,1)

	kd_m.update()
	kd_m.setParam('OutputFlag', False)
	#kd_m.write('lp.mps')
	kd_m.optimize()
	#print kd_m.Status
	try:
	    y_star = kd_m.objVal
	except:
	    #print "Error in y_star"
	    return Solution(-1* np.ones(160), -1, -1)
	#Continue with phase2, slot everyone- obj f_2
        C = load_obj('C')
	x = {} #reallocating memory
	bd_m = gp.Model()

        for o in Officers: 
	    for a in Assignments:
		x[(o,a)] = bd_m.addVar(vtype=gp.GRB.BINARY, obj=C[(o,a)],name="O{0:03d}".format(o) + "A{0:03d}".format(a))
		
        bd_m.ModelSense = gp.GRB.MINIMIZE #-1
	for a in Assignments:
            bd_m.addConstr(gp.quicksum(x[(o,a)] 
		for o in Officers),gp.GRB.EQUAL,1)	
	for o in Officers:
	    if o in list(restrictions):
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in A_r[o]), gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(directeds):
	        bd_m.addConstr(x[(o,A_d[o])], gp.GRB.EQUAL,1)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    elif o in list(rejects):
		bd_m.addConstr(x[(o,int(lpA[o]))], gp.GRB.EQUAL,0)
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	    else:
		bd_m.addConstr(gp.quicksum(x[(o,a)] for a in Assignments),gp.GRB.EQUAL,1)
	
	bd_m.addConstr(gp.quicksum(gp.quicksum(y[o]*x[(o,a)] for a in KD_Ass) for o in Officers), gp.GRB.LESS_EQUAL, y_star) 
	bd_m.setParam('OutputFlag', False)
	bd_m.update()
	bd_m.optimize()
	#print bd_m.Status
	#unassigned =[]
	sol.finalSolution = np.zeros(160)
	
        try:
	    for v in bd_m.getVars():
	        if v.x >0:
	            if int(v.varName[-3:])>=139 or int(v.varName[-3:]) in [71,72]:
	                #unassigned.append(int(v.varName[1:4]))
		        sol.finalSolution[int(v.varName[1:4])] = 999
		    else:		
		        sol.finalSolution[int(v.varName[1:4])] = int(v.varName[-3:])
	    sol.finalSolution = sol.finalSolution.astype(int)
	    sol.resultStatusFlag = 0 #Good Execution
	    #save_obj(sol.finalSolution, 'lpA') #saved locally first time and referenced later
	    sol.changes = sum(i != j for i, j in zip(sol.finalSolution, lpA))
	    sol.obj = bd_m.objVal
        except:
	    sol = Solution(-1* np.ones(160), -1, -1) 

    else: #Error
	print "Parameter Error: Invalid methodFlag"        
	sol = Solution(-1* np.ones(160), -1, -1)	

    if methodFlag == 1:
	    C = load_obj('C')
	    sol.obj = 0
	    for i,j in enumerate(sol.finalSolution):
		if j == 999:
		    pass
		else:
		    try:	
			sol.obj = sol.obj + C[(i,j)]
		    except KeyError: #infeasible solution
			print (i,j)
			sol.obj = -1


    	
    return sol

def eval(matching):
	C = load_obj('C')
	eval = 0
	for i,j in enumerate(matching):
	    if j == 999:
	        pass
	    else:
		try:	
		    eval = eval + C[(i,j)]
		except KeyError: #infeasible solution
		    sol.obj = -1
	return eval

def getRanks(matching):
        KD_Ass= range(0,71)
        B_Ass = range(73,139)
	r_kd = load_obj('r_kd')
	r_b  = load_obj('r_b')
	ranks = []
	for i,j in enumerate(matching):
	    if j == 999:
		pass
	    else:
		try:
		    if j in B_Ass:
			ranks.append(r_b[i][j-73])
		    elif j in KD_Ass:
			ranks.append(r_kd[i][j])
		    else:
			pass
		except KeyError:
		    ranks = [0]
	if len(ranks) >0:
	    return ranks
	else:
	    return [0]	


def bestOfN(N,aC):
	x={}
	bestObj = float("inf")
	bestPtr = 0
	for i in range(0,N):
		x[i] = SMA(aC)
		#if x[i].obj <bestObj and x[i].obj>0:
		if x[i].changes <bestObj and x[i].changes>-1:
		    bestPtr = i
		    bestObj = x[i].changes
	return x[bestPtr]


#If called from a command shell, call self
#if __name__ == "__main__":
#    import sys
#    assignmentFunction(sys.argv[1])
