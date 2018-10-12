# Thomas
My Bayes algorithm, for the name of Thomas Bayes

## Features
* Cope with continuous random varaibles intellegently.
* integer random varaibles (e.g. the mass of things with integer gram) will be treated as continuous ones in some case.

## Requirement
* numpy
* pandas
* scikit-learn (in examples)
* neupy (in gbayes.py)

## Why
For the Honor of T. Bayes

![](https://github.com/Freakwill/Thomas/blob/master/Thomas_Bayes.gif)


## Grammar
Just see the example file.

## Is it easy?
Yes

## Principle

### Naive Bayes


$$
p(c|x)=\frac{p(x|c)p(c)}{p(x)}\sim p(x|c)p(c)\\
\sim \prod_ip(x_i|c)p(c) = \prod_ip(x_i,c)p(c)^{1-n}~~~~~~~~~\text{(Naive condition)}\\
\sim\prod_i\frac{N(x_i,c)}{N}p(c)^{1-n}
$$


### Semi Naive Bayes

$$
p(c|x,y)=\sim p(x|c)p(c|y)\\
\sim \prod_ip(x_i|c)p(c|y) ~~~~~~~~~\text{(Semi-Naive condition)}
$$

When $y$ is empty, it is equiv. to the naive one.

## Predict

$$
\frac{p(c|x,y)}{p(c'|x,y)}= \prod_i(\frac{p(x_i|c)}{p(x_i|c')})\frac{p(c|y)}{p(c'|y)}\\
= \prod_i(\frac{p(x_i,c)}{p(x_i,c')})\frac{p(c|y)}{p(c'|y)}(\frac{p(c')}{p(c)})^n
~~~~~~~~~\text{(Semi-Naive condition)}\\
\sim \prod_i(\frac{N(x_i,c)}{N(x_i,c')})\frac{p(c|y)}{p(c'|y)}(\frac{N(c')}{N(c)})^n  ~~~~~~~~~~~~~~~~~~~~~\text{(estimate)}
$$



### 0-1 cases

$$
r = \frac{p(1|x,y)}{p(0|x,y)}\sim \prod_i(\frac{N(x_i,1)}{N(x_i,0)})\frac{p(c|y)}{1-p(c|y)}(\frac{N(0)}{N(1)})^n
$$

iff $r\geq 1$, $(x,y)$ is in class 1, else in class 0.



## Estimate (for continuous rv)

$p(x)\sim \frac{N(x)}{N}, N(x):$ the number of samples in a neighborhood of $x$


![](https://github.com/Freakwill/Thomas/blob/master/README.png)
