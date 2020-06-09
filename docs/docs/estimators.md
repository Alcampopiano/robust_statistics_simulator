## What is a trimmed mean?

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

