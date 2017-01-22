library(agricolae)
library(car)
library(caret)

A <- read.csv("~/Desktop/Thesis2/BestOf5-Obj/results.csv")
A <- read.csv("~/Desktop/Thesis2/Bestof10-Matches/results.csv")

A <- read.csv("~/Desktop/Thesis2/AnalysisMaster.csv")

A <- subset(A, changes >0 & feasible ==1 & totalpur >0)#AND <12!!
#A <- subset(A, feasible ==1)
#A <- subset(A, totalpur >0)

method <- factor(A$method)
obj <- as.numeric(A$obj)
res <- factor(A$restrictions)
dir <- factor(A$directeds)
rej <- factor(A$rejects)
chg <- as.numeric(A$changes)
time <- as.numeric(A$time)
rmean <- as.numeric(A$ranksmean)
rvar <- as.numeric(A$ranksvar)
 
m_chg <- aov(chg ~ method+res+dir+rej)
m_obj <- aov(obj ~ method+res+dir+rej)
m_time <- aov(time ~ method+res+dir+rej)
m_rmean <- aov(rmean ~ method+res+dir+rej)
m_rvar <- aov(rvar ~ method+res+dir+rej)


summary(m_obj)
scheffe.test(m_obj, "method", alpha=0.05, console=TRUE)
drop1(m_obj,~.,test="F")


summary(m_time)
scheffe.test(m_time, "method", alpha=0.05, console=TRUE)

summary(m_rmean)
scheffe.test(m_rmean, "method", alpha=0.05, console=TRUE)

summary(m_rvar)
scheffe.test(m_rvar, "method", alpha=0.05, console=TRUE)


summary(m_chg)
scheffe.test(m_chg, "method", alpha=0.05, console=TRUE)
t3m_chg <- drop1(m_chg,~.,test="F")

lm_chg <- lm(m_chg)
r_chg <-resid(lm_chg)
chg.press <- r_chg/(1-hat(model.matrix(lm_chg)))
plot(chg.press ~ method,ylim=range(chg.press))
#PRESS Stat is sum(chg.press^2)

r_stud_chg <-rstudent(lm_chg)
plot(r_stud_chg ~method, ylim=range(r_stud_chg))


#now just look at method == {2, 1Lex, 1-5m}
A <- subset(A, changes >0 & feasible ==1 & totalpur<12 & totalpur >0 & (method=="2"|method=="1Lex"|method=="1-5m"))

#> sum(chg.press^2)
#[1] 193343 #MUCH BETTER!!

#? Model as numeric?

res <- as.numeric(A$restrictions)
dir <- as.numeric(A$directeds)
rej <- as.numeric(A$rejects)

outlierTest(lm_chg)
qqPlot(lm_chg)
leveragePlots(lm_chg)


#influence plot
influencePlot(lm_chg, id.method="identify", main="Influence Plot", sub="Circle size is proportial to Cook's Distance" )

ncvTest(lm_chg) #Breusch–Pagan, reject null of homosced.

chg_transf <- BoxCoxTrans(A$changes)




vif(lm_chg) #no multicollinearity

plot(hatvalues(lm_chg), main = "Hat Values for Total Changes(Reuced Model")





















