# Estimators used in the simulator
The following list describes each estimator used in the robust statistics simulator.

## Sample mean
The sample mean (arithmetic) is the sum of values divided by the number of values. In symbols,
this can be expressed as follows:
$$
\bar{X} = \frac{X_i\,+,...,+\,X_n}{n}
$$

## Trimmed mean
The trimmed mean involves calculating the sample mean after
removing a proportion of values from each
tail of the distribution. In symbols the trimmed mean is expressed as
follows:

$$
\bar{X}_t = \frac{X_{(g+1)}\,+,...,+\,X_{(n-g)}}{n-2g}
$$

where $X_1, \,X_2,\,...\,,X_n$ is a random sample and
$X_{(1)}, \le X_{(2)}\,,...,\,\le X_{(n)}$ are the observations in
ascending order. The proportion to trim is $\gamma\,(0\lt \gamma \lt.5)$
and $g = [ \gamma n ]$ rounded down to the nearest integer.

## Median
Given a finite list of numbers ordered from smallest to largest, the median is
the "middle" number. When there are an even number of values the median is typically taken as
the average between the two "middle" values. The median can be expressed as follows:

$$
X_M = \frac{1}{2}(X_{\lfloor (n+1)/2 \rfloor} +  X_{\lceil (n+1)/2 \rceil})
$$

where $X$ is an ordered list of $n$ numbers. $\lfloor \rfloor$ and $\lceil \rceil$
denote rounding to the floor or ceiling integers, respectively.