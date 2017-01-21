library(agricolae)

A <- read.csv("~/Desktop/Thesis2/BestOf5-Obj/results.csv")
A <- read.csv("~/Desktop/Thesis2/Bestof10-Matches/results.csv")

A <- subset(A, changes >0)
A <- subset(A, feasible ==1)

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


summary(m_chg)
scheffe.test(m_chg, "method", alpha=0.05, console=TRUE)

summary(m_obj)
scheffe.test(m_obj, "method", alpha=0.05, console=TRUE)

summary(m_time)
scheffe.test(m_time, "method", alpha=0.05, console=TRUE)

summary(m_rmean)
scheffe.test(m_rmean, "method", alpha=0.05, console=TRUE)

summary(m_rvar)
scheffe.test(m_rvar, "method", alpha=0.05, console=TRUE)


