install.packages("ggplot2")
install.packages("dplyr")

#loading the dataset 
df=read.csv("C:\\Users\\Abishek\\Documents\\SEM_5\\5_DA\\Project\\zomato.csv")

# PRE-PROCESSING ########################


#removing columns: url, phone, dish_like and menu_item 
dfcl=df[-c(1,8,11,15)]

#identifying empty strings and removing them
dfcl[dfcl==""]<-NA
dfcl=na.omit(dfcl)

#Removing the '/5' from rating columns
dfcl$rate=lapply(dfcl$rate ,function(x) gsub("/5", "",x) )

#converting rate to numeric
dfcl$rate=as.numeric(dfcl$rate)

#Converting output classes "Yes", "No" to 0 and 1

dfcl$book_table=as.character(dfcl$book_table)
dfcl$book_table[dfcl$book_table=="Yes"]=1
dfcl$book_table[dfcl$book_table=="No"]=0
dfcl$book_table=as.factor(dfcl$book_table)

dfcl$online_order=as.character(dfcl$online_order)
dfcl$online_order[dfcl$online_order=="Yes"]=1
dfcl$online_order[dfcl$online_order=="No"]=0
dfcl$online_order=as.factor(dfcl$online_order)

#removing unusual characters from the name columns
dfcl$name=iconv(dfcl$name,"latin1","ASCII",sub="")

#removing commas from the cost columns
dfcl$approx_cost.for.two.people.=lapply(dfcl$approx_cost.for.two.people. ,function(x) gsub(",", "",x) )

#converting cost to numeric columns:
dfcl$approx_cost.for.two.people.=as.numeric(dfcl$approx_cost.for.two.people.)

#############################################################################



newdata <- dfcl[order(-dfcl$votes),]
dfsorted=distinct(newdata[-12], newdata$name, .keep_all = TRUE)
dfsorted=dfsorted[1:10,]

ggplot(dfsorted) + 
  geom_bar(aes(x = dfsorted$name,y=dfsorted$votes),  stat = 'identity', position = position_dodge(preserve = 'single'))+
  #theme(axis.text.x = element_text(angle = 45, hjust = 1,size = 7))+
  coord_flip()+ labs(title = "Top 10 Most popular restaurants", y= "Votes", x = "Restaurant")

c=count(dfcl,dfcl$name)
c=c[order(-c$n),]
c=c[1:10,]
colnames(c)=c("name","n")
ggplot(c) + 
  geom_bar(aes(x =c$name,y=c$n),  stat = 'identity', position = position_dodge(preserve = 'single'))+
  #theme(axis.text.x = element_text(angle = 45, hjust = 1,size = 7))+
  coord_flip()+ labs(title = "Top 10 Most number of outlets", y= "No. of outlets", x = "Restaurant")








#######################################################

qplot(dfcl$rate, 
      geom="histogram",
      main = "Histogram for Rating distribution", 
      xlab = "Rating",
      ylab="Count",
      col=I("red"), 
      alpha=I(.2))



qplot(dfcl$votes, 
      geom="histogram",
      main = "Histogram for Votes distribution", 
      xlab = "Votes",
      ylab="Count",
      col=I("red"), 
      alpha=I(.2))

qplot(dfcl$approx_cost.for.two.people., 
      geom="histogram",
      main = "Histogram for Cost distribution", 
      xlab = "Cost for 2",
      ylab="Count",
      col=I("red"), 
      alpha=I(.2))

###################################################



s=subset(dfcl,book_table==0)
l1=length(s$book_table)
s=subset(dfcl,book_table==1)
l2=length(s$book_table)
val=c("Yes","No")
d=data.frame(val,c(l2,l1))
colnames(d)=c("key","val")

mycol <- c("#0073C2FF", "#EFC000FF", "#868686FF", "#CD534CFF")

ggplot(d, aes(x = "", y = val, fill = key)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  geom_text(aes(y = d$val, label = val), color = "black")+
  ggtitle("Restaurants allowing reservations")+
  scale_fill_manual(values = mycol) +
  theme_void()

s=subset(dfcl,online_order==0)
l1=length(s$online_order)
s=subset(dfcl,online_order==1)
l2=length(s$online_order)
val=c("Yes","No")
d=data.frame(val,c(l2,l1))
colnames(d)=c("key","val")

mycol <- c("#0073C2FF", "#EFC000FF", "#868686FF", "#CD534CFF")

ggplot(d, aes(x = "", y = val, fill = key)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  geom_text(aes(y = d$val, label = val), color = "black")+
  ggtitle("Restaurants with online orders")+
  scale_fill_manual(values = mycol) +
  theme_void()
#####################################################################
plot(dfcl$rate, dfcl$votes, col="blue", cex=0.5, xlab="Rating", ylab="Votes", main="Distribution of votes with rating")
plot(dfcl$approx_cost.for.two.people., dfcl$votes, col="blue", cex=0.4, xlab="Approx. cost for two people", ylab="Votes", main="Distribution of cost with votes")

