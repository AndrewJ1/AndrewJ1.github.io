---
layout: post
title:      "Choosing Research Topics"
date:       2019-09-11 20:10:20 +0000
permalink:  choosing_research_topics
---


As a data scientist you will get open ended questions from stakeholders. 

“Take a look at the data, and give me some recommendations”

But where do you begin your research? The good news is that there’s no one perfect way to do things. 


In this blog I will cover how I went about determining my research topics for a project using the Northwind database (an open-source dataset created by Microsoft, containing data from a fictional company). The goal of this project is to query the database to get the data needed to perform a statistical analysis. In this statistical analysis, I need to perform hypothesis tests to answer four different research questions.


The data that we have access to, is illustrated in the Entity Relationship Diagram (ERD) of the database.

![](https://i.imgur.com/yj8IgCh.png)


In making an observation normally there would be existing research or a Database administrator that I could talk to. In this scenario, all I have is the database to explore.

So what to test? We need to ask what the business cares about. What is the business situation right now? How could we improve the business?

Another consideration: who is your audience? For my project, this is open ended. I am choosing to present to Northwind management.

In picking topics to research I must consider the needs of my audience and tune my message to meet these needs. To have my ideas and research get through to Northwind management they need to stand out. Even facts need some emotional appeal where we introduce a story behind them. 

We also need to consider whether we should focus on a broad view, or a deep view of the problem. A deep view would have us researching further and further into one aspect – promotions for example, while a broad view looks across all aspects of the business.

At this point, I would have a lot of questions for whoever had requested the research. 

Questions like: 

“What is the business problem we are trying to solve with the study?”

“Who is the analysis for?”

“How will the research be used?”

“What should I prioritize?”

“Has any work been done already?”


Once started, I want to quickly return some initial findings to the requestor, and will have a set of new questions. This will keep me on the right track, and allow for redirection if the results are not completely aligned with what the requestor really wanted. It will also help with managing expectations. We’ll have a fair idea of whether the data is robust enough to provide the results we are looking for, and if not we can reevaluate or pull the plug on the research.


For this project, I have chosen to go with a broad view. This will give me a better understanding of the company as a whole. It will mean leaving some unanswered questions about detail that I can come back to later. There are always more questions to be asked, and things to discover, so part of my findings will be presenting options for future work.


The topics that I will be looking into are:

•	Promotions – What is the effect of discounts on order quantity?

•	Products – Is there a different in order quantity by product group?

•	Employees – There are 2 main sales offices. Is there any difference between them? Is there any difference in performance by sales representation?

•	Customers – Here I had to re-evaluate. I was going to look into customer demographics, but found out that table was empty. I will instead research is there any difference in revenue by customer region?


Now that my topics are chosen, I’ll take you through my analysis of the promotions topic.


SQL query to return the data

```
cur.execute("""SELECT *,
                    CASE 
                    WHEN od.Discount == 0 THEN 0
                    ELSE 1 
                    END AS 'Disc_0_1'
                FROM [OrderDetail] od;""")
ord_det = pd.DataFrame(cur.fetchall())
ord_det.columns = [i[0] for i in cur.description]
ord_det.head()

```


Violin plot to view the analysis

```
sns.set(rc={'figure.figsize':(10,8)})
sns.set(color_codes=True)
sns.violinplot( x=ord_det["Disc_Group"], y=ord_det["Quantity"], linewidth=2, palette="Blues")
plt.xlabel("Discount Groups")
plt.ylabel("Quantity")
plt.title('Comparing Quantity Distribution by Discount Group', fontsize = 16)
plt.show();

```

![](https://i.imgur.com/FexnRRY.png)



Interpretation.  All the discount groups have a higher mean and interquartile range than the non-discount group. The discount groups have a flatter, more spread distribution than non-discount; especially at 20% and 25% discount. The 10% has a similar mean to 5% and 15%, but the interquartile range appears to be lower and of a smaller range.



**Welch's t-test**
```
Welch_ttest = stat.ttest_ind(No_Discount_Qty, Discount_Qty, equal_var=False)
print(Welch_ttest)
```


Interpretation
We get a p-value of 2.8282e-10, this is less than our 𝛼=0.05 
Since 𝑝<𝛼  we reject the Null Hypothesis that discount has no effect on order quantity. There is statistical evidence at the 5% level of significance that the mean order quantity of discounted orders is greater than the mean order quantity with no discount.



**Cohen's d**
```
def Cohen_d(group1, group2):
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled variance
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d

```


Interpretation
Cohen provided some general “rule of thumb” guidelines (which should be used cautiously):

Small effect = 0.2 Medium Effect = 0.5 Large Effect = 0.8

Based on these guidelines we can say that discounts have a small statistically significant effect on order quantity



**ANOVA**
```
f1 = 'Disc_Group'
f2 = 'Quantity'

formula = '{} ~ C({})'.format(f2, f1)
lm = ols(formula, ord_det).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```


 PR(>F)
1.816734e-08


Interpretation
With ANOVA we are testing whether there is a significant difference between discount groups. The model gives us a p-value of 1.8167e-08, which is less than our 𝛼=0.05α=0.05. We can reject the Null Hypothesis that there is no difference between groups. We don't know which specific levels of discount are significant, only that in total they are significant.



**Tukey**

```
mc = MultiComparison(ord_det.Quantity, ord_det.Disc_Group)
result = mc.tukeyhsd()
 
print(result)
print(mc.groupsunique)
```


![](https://i.imgur.com/9IoAdMp.png)
​ 


Interpretation
Discount levels of 5%, 15%, 20% and 25% produce a significant difference compared against no discount. However, there is no significant difference between the different levels of discount.

This is a really interesting result, and there is a story here. 

Why is 10% not a significant result? In what ways is it different from the other discount groups – product, customer, pricing?

In addition it is interesting that there is no difference between discount levels – only discounted vs non-discounted. On the surface this would suggest that Northwind should stop higher level discounts, and use only 5% as there is no extra sales benefit from the increased discount. However, there may be well be other differences between the groups that we are yet to discover.

And these are questions for our future work summary



