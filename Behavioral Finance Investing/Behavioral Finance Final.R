### Stage One: What's your trading strategy ###
# This program illustrates how to replicated a long-short strategy
# based on stocks maximum daily return (MAX) following Bali et al. (2011)
# Our purpose is to create a MAX portfolio trading strategy

### Stage Two: form the portfolio ###
# STEP 1: read the data
daily_stock=read.csv(file.choose(),header=TRUE) # read the stock return data
daily_stock=daily_stock[,-1]
head(daily_stock) # show  the first several rows for daily_stock
dim(daily_stock) # check the size of matrix
colnames(daily_stock) # show the names for each column
str(daily_stock)

daily_stock$date=as.Date.factor(daily_stock$date)
range(daily_stock$date)

# STEP 2: calculate MAX as the maximum daily return for each stock within the previous month
daily_stock$year=substr(daily_stock$date,1,4) #get the year for each observation
daily_stock$month=substr(daily_stock$date,6,7) #get the year for each observation
daily_stock$umonth=12*(as.numeric(daily_stock$year)-as.numeric(min(daily_stock$year)))+
  as.numeric(daily_stock$month) # get the unique month id for each observation

# get the max return
# approach 1: using aggregate function
max(daily_stock$ret[daily_stock$permno==10006 & daily_stock$umonth==1])

start.time=Sys.time() # record the start time for this program

max_ret=aggregate(ret~permno+umonth,daily_stock,max)

end.time=Sys.time() # record the end time for this program

end.time-start.time # the time taken to run this program
rm(end.time,start.time)

# approach 2: using dplyr package
# 量化程序需要精简时间
install.packages("dplyr") # install the package
library(dplyr) # load the installed package

start.time=Sys.time() # record the start time for this program

# Let’s create a dataframe "max_ret" of aggregated data from the "daily_stock" dataset.
# I’ll group the data according to the columns "permno" and "umonth".
# I’ll then create summary statistic of the maximum ret across each grouping.  # filter(n()>=15)  %>% # Filter condition
max_ret=daily_stock %>% # Specify original dataframe
  group_by(permno,umonth) %>% # Grouping variable(s)
  summarise(max=max(ret)
            ) # we would like to select for each firm in each month the largest daily return
dim(max_ret)
max_ret=as.data.frame(max_ret) # convert tibble to data.frame
dim(max_ret)

end.time=Sys.time() # record the end time for this program

end.time-start.time # the time taken to run this program
rm(end.time,start.time)


# STEP3: we would like to long (short) stocks with high (low) MAX in the PREVIOUS month t-1.
# and hold the long-short portfolio for one month (month t), until we rebalance the portfolio at the end of the month (t).
# in this step, we need monthly stock return data
monthly_stock=read.csv(file.choose(),header=TRUE) # download the monthly stock return data
monthly_stock=monthly_stock[,-1] # delete the first column which is useless
monthly_stock$year=substr(monthly_stock$date,1,4) #get the year for each observation
monthly_stock$month=substr(monthly_stock$date,6,7) #get the year for each observation
monthly_stock$umonth=12*(as.numeric(monthly_stock$year)-as.numeric(min(monthly_stock$year)))+as.numeric(monthly_stock$month) # get the unique month id for each row
dim(monthly_stock)
monthly_stock$mkt=monthly_stock$shrout*monthly_stock$prc # we also calculate the market capitalization for each stock in each month

# STEP 4: to sort stocks based on their MAX in the last month, we merge this month's return with last month's MAX
monthly_stock$umonth_pre=monthly_stock$umonth-1 # get the umonth in the previous month
dim(monthly_stock)
monthly_stock2=merge(monthly_stock,max_ret,
                  by.x=c("permno","umonth_pre"),
                  by.y=c("permno","umonth"))
dim(monthly_stock2) # Create a new matrix "monthly_stock2" by merging last month's MAX with this month's return for each stock

monthly_stock2=merge(monthly_stock2,monthly_stock[,c(1,8,9)],
             by.x=c("permno","umonth_pre"),
             by.y=c("permno","umonth"))
dim(monthly_stock2) # merge last month's mkt with this month's return

# STEP 5: We form decile portfolios in an ascending order of MAX in the last month
monthly_stock2$xproxy=monthly_stock2$max
monthly_stock2$yproxy=monthly_stock2$ret

# matrix "cor" contains the monthly returns for each decile portfolio (on row) in each month (on column)
# There are two loops in the "for" function
# in the outside loop (for i in 1:max(monthly_stock$umonth)), we rebalance the portfolio in each month
# in the inside loop (for j in 1:10), we sort the stocks into 10 deciles based on xproxy (MAX)
# to be more specific, if j=1, stocks are those in the lowest MAX decile
# if j=10, stocks are those in the highest MAX decile
# and we calculate the return for each portfolio
n=max(monthly_stock2$umonth)-min(monthly_stock2$umonth)+1 # calculate how many months you have in your sample
#return_equal=matrix(0,nrow=10,ncol=n) # prepare a matrix for equal-weighted portfolio
return_mkt=matrix(0,nrow=10,ncol=n) # prepare a matrix for value-weighted portfolio

for (i in 1:n){
  umonth=sort(unique(monthly_stock2$umonth))[i]
  test_sub=monthly_stock2[which(monthly_stock2$umonth==umonth),]
  for (j in 1:10)
  {
   #return_equal[j,i]=mean(test_sub$yproxy[which(test_sub$xproxy>=quantile(test_sub$xproxy,(j-1)*0.1,na.rm=TRUE)
                                       # &test_sub$xproxy<=quantile(test_sub$xproxy,j*0.1,na.rm=TRUE))],
                     # na.rm=TRUE) # equal-weighted portfolio
    return_mkt[j,i]=weighted.mean(test_sub$yproxy[which(test_sub$xproxy>=quantile(test_sub$xproxy,(j-1)*0.1,na.rm=TRUE)
                                                 &test_sub$xproxy<=quantile(test_sub$xproxy,j*0.1,na.rm=TRUE))],
                           test_sub$mkt.y[which(test_sub$xproxy>=quantile(test_sub$xproxy,(j-1)*0.1,na.rm=TRUE)
                                                 &test_sub$xproxy<=quantile(test_sub$xproxy,j*0.1,na.rm=TRUE))],
                           na.rm=TRUE) # value-weighted portfolio, where the weight is determined by the market capitalization
  }
  print(umonth)
}



### Stage Three: evaluate the portfolio ###
# STEP 6: We obtain a time-series returns, and run t-test
return_equal=t(return_equal) # transform the matrix
return_equal=as.data.frame(return_equal)
colnames(return_equal)=c("q1","q2","q3","q4","q5","q6","q7","q8","q9","q10") # name each column
return_equal$longshort=return_equal$q1-return_equal$q10 # this is the time-series return for your long-short strategy
t.test(return_equal$longshort) # we run t-test to see if this portfolio's return is significantly different from zero

return_mkt=t(return_mkt) # transform the matrix
return_mkt=as.data.frame(return_mkt)
colnames(return_mkt)=c("q1","q2","q3","q4","q5","q6","q7","q8","q9","q10") # name each column
return_mkt$longshort=return_mkt$q1-return_mkt$q10 # this is the time-series return for your long-short strategy
t.test(return_mkt$longshort) # we run t-test to see if this value-weighted portfolio's return is significantly different from zero

# plot the cumulative return
return_mkt$cum_ret=return_mkt$longshort+1
for (i in 1:nrow(return_mkt)){
  return_mkt$cum_ret2[i]=prod(return_mkt$cum_ret[1:i])
}

plot(return_mkt$cum_ret2)


# STEP 7: We adjust raw returns with benchmark and calculate the alphas of the strategy
market_return=read.csv(file.choose(),header=TRUE)
colnames(market_return)[1:5]=c("yearmon","mkt_rf","SMB","HML","RF")

range(monthly_stock2$umonth)
monthly_stock2$yearmon=as.numeric(monthly_stock2$year)*100+as.numeric(monthly_stock2$month)
range(monthly_stock2$yearmon)

market_return=market_return[which(market_return$yearmon>=min(monthly_stock2$yearmon) &
                                    market_return$yearmon<=max(monthly_stock2$yearmon)
                                    ),] # we get the market return at the same time period
return_mkt=cbind(return_mkt,market_return[,2])
colnames(return_mkt)[ncol(return_mkt)]="mkt_rf"
return_mkt=cbind(return_mkt,market_return[,3])
colnames(return_mkt)[ncol(return_mkt)]="SMB"
return_mkt=cbind(return_mkt,market_return[,4])
colnames(return_mkt)[ncol(return_mkt)]="HML"
return_mkt=cbind(return_mkt,market_return[,5])
colnames(return_mkt)[ncol(return_mkt)]="RF"

# we regress the monthly longshort portfolio return on the contemporaneous monthly market return
# the intercept that we get is the alpha of such strategy
regression <- lm(return_mkt$longshort~return_mkt$mkt_rf+return_mkt$SMB+return_mkt$HML+return_mkt$RF,
                 data=return_mkt)
summary(regression) 


# STEP 8: We evaluate the performance of the strategy in different sentiment levels
sentiment=read.csv(file.choose(),header=TRUE)
colnames(sentiment)=c("yearmon","sent1","sent2")
# we get the sentiment data in the last month
# since we form the portfolio at the end of each month t-1
# based on MAX observation in month t-1
# and sell the portfolio at the end of month t
# we need the sentiment data in month t-1 for each month t's portfolio return

sentiment$high=0
sentiment$high[which(sentiment$sent1>=median(sentiment$sent1))]=1
sentiment$high[which(sentiment$sent1<median(sentiment$sent1))]=0

t.test(return_mkt$longshort[which(sentiment$high==1)])
t.test(return_mkt$longshort[which(sentiment$high==0)])

test=return_mkt[which(sentiment$high==1),]
regression <- lm(test$longshort~test$mkt_rf+test$SMB+test$HML+test$RF, data=test)
summary(regression) 

test=return_mkt[which(sentiment$high==0),]
regression <- lm(test$longshort~test$mkt_rf+test$SMB+test$HML+test$RF, data=test)
summary(regression) 


