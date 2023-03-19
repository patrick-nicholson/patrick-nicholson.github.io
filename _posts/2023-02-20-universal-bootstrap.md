---
layout: post
author: "Patrick Nicholson"
title: "Universal bootstrap: a superpower"
excerpt: "Bootstrapping is commonly used in computational statistics and machine learning for uncertainty quantification, hypothesis testing, and ensembling. By combining universal sampling with Poisson bootstrap, I show how the universal bootstrap unlocks incredibly sophisticated analysis at any scale in any tool."
notebook:
    path: /notebooks/universal-bootstrap.ipynb
image:
    path: /notebooks/universal-bootstrap_files/universal-bootstrap_30_0.png
---

> _Look on my computational methods, ye theorists, and despair_

[Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) is a commonly used technique for computational statistics and machine learning. 
* [Uncertainty quantification](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)): an approximate distribution of a sample statistic (e.g., mean) is the empirical distribution of the same statistic calculated over bootstrap samples
* [Bootstrap tests](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bootstrap_hypothesis_testing): a distribution-free hypothesis test by using the empirical distribution of the test statistic from bootstrap samples that randomize the test design
* [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating): improving accuracy and reducing variance of a learner by training the model on subsets of resampled 

So how _should_ you bootstrap? In an earlier post, I covered [universal sampling](https://patrick-nicholson.github.io/2023/02/13/universal-sampling/) and its advantages. In this post, I show how this can be extended with sampling with replacement to create the universal bootstrap.


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from IPython.display import Markdown
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.utils.murmurhash import murmurhash3_32
from uuid import uuid4

%matplotlib inline

random_state = np.random.RandomState(12345)
```


```python
INT_MIN = np.int32(-(2**31))
INT_MAX = np.int32(2**31 - 1)
INT_RANGE = np.int64(2**32)

_mmhash_ufunc = np.frompyfunc(murmurhash3_32, nin=1, nout=1)


def mmhash(values):
    """Wrapper for sklearn's MurmurHash that accepts most types"""
    if np.ndim(values):
        if np.issubdtype(values.dtype, np.int32):
            return murmurhash3_32(values)
        return _mmhash_ufunc(values.astype(np.bytes_)).astype(
            np.int32
        )
    if isinstance(values, (bytes, str, np.int32)):
        return murmurhash3_32(values)
    return _mmhash_ufunc(np.array(values, dtype=np.bytes_))
```

## Key insight

Bootstrapping is based on sampling with replacement. [Poisson sampling](https://en.wikipedia.org/wiki/Poisson_sampling) is one such method. For a sample size $N$ and resampling size $S$, the resampling weights for each observation follows the $\text{Pois} \left( \frac{S}{N} \right)$ distribution. The [Poisson bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Poisson_bootstrap), commonly used for bootstrapping streams and large datasets, is the particular case where $N = S$, i.e., $\text{Pois}(1)$.

In [my earlier post](https://patrick-nicholson.github.io/2023/02/13/universal-sampling/), I demonstrated a method of sampling from a Poisson distribution based on universal hash functions. Composing this with Poisson sampling gives us the universal bootstrap.

In short:
* A universal hash function deterministically maps an input to a uniformly distributed integer in the full integer range
* Uniform integers can be transformed into Poisson values
* Multiplying the hash value by another random integer yields an uncorrelated random integer
* Ergo, mapping the products of a hash value by $r$ random integers to $\text{Pois}(1)$ values provides $r$ deterministic bootstrap sample weights


```python
def poisson_thresholds(lam, tol=None):
    """Threshold values (32-bit integers) for Poisson inverse 
    transformation
    """

    tol = tol or 1e-6
    pois = stats.poisson(lam)
    lower, upper = np.ceil(pois.isf([1 - tol, tol])).astype(
        np.int32
    )
    edges = (
        pois.cdf(np.arange(lower, upper)) * 2**32 - 2**31
    ).astype(np.int32)

    # compress int.min
    lower += (edges == -(2**31)).sum()
    edges = edges[edges > -(2**31)]

    # compress int.max
    edges = np.r_[edges[edges < 2**31 - 1], 2**31 - 1]
    upper = lower + len(edges) - 1

    return lower, upper, edges


def inverse_transform_search(thresholds, hash_values):
    """Inverse transform search with correction to thin out tails
    when there are duplicated thresholds
    """
    left = np.searchsorted(thresholds, hash_values, side="left")
    right = np.searchsorted(
        thresholds, hash_values, side="right"
    )
    return np.where(hash_values < 0, right, left)


def poisson_sample_weight(hashes, lam, tol=None):
    """Poisson weights for an array of hashes"""
    lower, _, edges = poisson_thresholds(lam, tol)
    return inverse_transform_search(edges, hashes) + lower
```

## Baby's first bootstrap

The first thing a little data scientist learns to do with a bootstrap is to estimate a distribution around some sample statistic. Let's use our old friend `iris` for this.


```python
bootstrap_replications = 200

_ = load_iris(as_frame=True)
iris = pd.concat([_["target"], _["data"]], axis=1)
iris.columns = ["_".join(k.split()[:2]) for k in iris]

iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



We use this to estimate some sample statistic.


```python
iris["sepal_length"].mean()
```




    5.843333333333334



A single bootstrap estimate is the same statistic applied to a resampled from the data. The empirical distribution of many different bootstrap replications is the bootstrap distribution of the statistic.


```python
iris["sepal_length"].sample(frac=1.0, replace=True).mean()
```




    5.796666666666667




```python
n = len(iris)

baby_null = (
    iris["sepal_length"]
    .sample(
        frac=bootstrap_replications,
        replace=True,
        ignore_index=True,
    )
    .groupby(lambda i: i // n)
    .mean()
)

fig, ax = plt.subplots()
ax.hist(baby_null)
ax.axvline(iris["sepal_length"].mean(), c="red")
ax.set_title("Bootstrap distribution of a sample statistic");
```


    
![png](/notebooks/universal-bootstrap_files/universal-bootstrap_12_0.png)
    


A single universal bootstrap estimate is the sample statistic weighted by Poisson weights. The universal bootstrap distribution repeats this by randomizing the hashes and repeating the weighting process.


```python
baby_hashes = mmhash(iris.index.values)
np.average(
    iris["sepal_length"],
    weights=poisson_sample_weight(baby_hashes, 1),
)
```




    5.844594594594595




```python
randomization = random_state.randint(
    INT_MIN, INT_MAX, bootstrap_replications, dtype=np.int32
)
baby_weights = poisson_sample_weight(
    np.multiply.outer(baby_hashes, randomization), 1
)
poisson_baby_null = (
    baby_weights * iris["sepal_length"].values[:, None]
).sum(axis=0) / baby_weights.sum(axis=0)

fig, ax = plt.subplots()
ax.hist(poisson_baby_null)
ax.axvline(iris["sepal_length"].mean(), c="red")
ax.set_title(
    "Universal bootstrap distribution of a sample statistic"
);
```


    
![png](/notebooks/universal-bootstrap_files/universal-bootstrap_15_0.png)
    


Of course, you can also do this by repeating observations based on the Poisson weights.


```python
values = np.tile(iris["sepal_length"], bootstrap_replications)
baby_weights = baby_weights.T.ravel()
replications = np.repeat(
    np.arange(bootstrap_replications), len(iris)
)
values, baby_weights, replications = (
    values[baby_weights > 0],
    baby_weights[baby_weights > 0],
    replications[baby_weights > 0],
)

repeated_estimates = (
    pd.Series(np.repeat(values, baby_weights))
    .groupby(np.repeat(replications, baby_weights))
    .mean()
)

fig, ax = plt.subplots()
ax.hist(repeated_estimates)
ax.axvline(iris["sepal_length"].mean(), c="red")
ax.set_title(
    "Universal bootstrap distribution of a sample statistic by"
    " repetition"
);
```


    
![png](/notebooks/universal-bootstrap_files/universal-bootstrap_17_0.png)
    


## Universal bootstrap for hypothesis testing

While baby's first bootstrap was fun and all, the practical power of bootstrapping is revealed when estimating the $H_0$ distribution of a hypothesis test. 

Given a sample of $N = n^T + n^C$ units with $n^T$ test units and $n^C$ control units, a single bootstrap iteration samples (with replacement) pseudo-test and pseudo-control groups; each pseudo group is (approximately) the same size as the true group. That is, we randomly assign (with replacement) each unit to the pseudo groups. The iteration's sample statistic (e.g., difference in means) is then calculated from the pseudo groups.

Let's simulate as simple experiment where an offer creates an average standardize effect of .1 on purchases.


```python
# number of customers
num_customers = 10**5

# percent of customers that received an offer
test_percent_offer = 0.35

# true effect of offer on standardized purchases
true_effect_offer = 0.1

# purchases across customers
# - customer_uuid: a UUID4 for the customer
# - customer_numeric_id: a serial ID for the customer
# - received_offer: customer received an offer (binary)
# - purchases: standardized purchases
purchases = pd.DataFrame(
    [
        (
            str(uuid4()),
            numeric_id,
            received_offer,
            purchases + received_offer * offer_effect,
        )
        for numeric_id, (
            received_offer,
            purchases,
            offer_effect,
        ) in enumerate(
            zip(
                random_state.binomial(
                    1, test_percent_offer, num_customers
                ),
                random_state.normal(0, 1, num_customers),
                random_state.normal(
                    true_effect_offer,
                    true_effect_offer,
                    num_customers,
                ),
            )
        )
    ],
    columns=[
        "customer_uuid",
        "customer_numeric_id",
        "received_offer",
        "purchases",
    ],
)
```

The universal bootstrap is changed only slightly: we now generate test and control weights according to their sample sizes. Each replication has pseudo-test and pseudo-control samples, from which we calculate the test value.


```python
test_randomization, control_randomization = random_state.randint(
    INT_MIN, INT_MAX, (2, bootstrap_replications), np.int32
)
```

The test is then comparing the null distribution to the observed statistic from the true test and control groups.


```python
offer_hashes = mmhash(purchases["customer_uuid"].values)

test_weights = poisson_sample_weight(
    np.multiply.outer(offer_hashes, test_randomization),
    test_percent_offer,
)
test = (test_weights.T * purchases["purchases"].values).sum(
    axis=1
) / test_weights.sum(axis=0)
del test_weights

control_weights = poisson_sample_weight(
    np.multiply.outer(offer_hashes, control_randomization),
    1.0 - test_percent_offer,
)
control = (
    control_weights.T * purchases["purchases"].values
).sum(axis=1) / control_weights.sum(axis=0)
del control_weights

offer_null_distribution = test - control

control, test = purchases.groupby("received_offer")[
    "purchases"
].mean()
offer_point_est = test - control

fig, ax = plt.subplots()
ax.hist(offer_null_distribution)
ax.axvline(offer_point_est, color="red", ls="--")
ax.set_title(
    "Universal bootstrap test: null distribution vs. observed"
    " difference"
);
```


    
![png](/notebooks/universal-bootstrap_files/universal-bootstrap_23_0.png)
    


What if we have a more complicated experiment? Consider an educational intervention:
* A test is taken twice
* An intervention is applied to some classes after the first test
* Classes are not identical: they have different average test scores
* Test scores improve over time 

This is intentionally set up for a difference-in-differences design.


```python
# number of classes
num_classes = 100

# average number of students in a class
avg_students = 25

# average test scores differ between classes
between_class_sd = 0.1

# students improve on the test over time
true_trend = 0.1

# percent of classes receiving an intervention
test_percent_class = 0.75

# true effect of the intervention on standardized test scores
true_effect_intervention = 0.05

# test scores in two periods for students within classes
# - class_id: unique class identifier
# - student_num: anonymous student number within class
# - intervention_class: class received intervention (binary)
# - post_period: score is for the second test (post intervention, if received)
# - score: standardized test score
test_scores = pd.DataFrame(
    [
        (
            class_id,
            student_num,
            intervention_class,
            post_period,
            start_score
            + post_period * trend
            + post_period
            * intervention_class
            * intervention_effect,
        )
        for class_id, (
            students,
            class_avg_score,
            intervention_class,
        ) in enumerate(
            zip(
                random_state.poisson(avg_students, num_classes),
                random_state.normal(
                    0, between_class_sd, num_classes
                ),
                random_state.binomial(
                    1, test_percent_class, num_classes
                ),
            )
        )
        for student_num, (
            start_score,
            intervention_effect,
        ) in enumerate(
            zip(
                random_state.normal(
                    class_avg_score, 1, students
                ),
                random_state.normal(
                    true_effect_intervention,
                    true_effect_intervention / 2.8,
                    students,
                ),
            )
        )
        for post_period, trend in enumerate(
            random_state.normal(true_trend, true_trend / 2.8, 2)
        )
    ],
    columns=[
        "class_id",
        "student_num",
        "intervention_class",
        "post_period",
        "score",
    ],
)
```

Correct measurement requires correct specification of standard errors. Standard covariances are too large to draw the correct inference; clustered standard errors are necessary.


```python
fit = smf.ols(
    "score ~ intervention_class * post_period", test_scores
).fit()
fit.summary(slim=True)
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>score</td>   <th>  R-squared:         </th> <td>   0.006</td>
</tr>
<tr>
  <th>Model:</th>               <td>OLS</td>    <th>  Adj. R-squared:    </th> <td>   0.005</td>
</tr>
<tr>
  <th>No. Observations:</th>  <td>  5034</td>   <th>  F-statistic:       </th> <td>   10.26</td>
</tr>
<tr>
  <th>Covariance Type:</th>  <td>nonrobust</td> <th>  Prob (F-statistic):</th> <td>9.91e-07</td>
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>    0.0911</td> <td>    0.040</td> <td>    2.277</td> <td> 0.023</td> <td>    0.013</td> <td>    0.170</td>
</tr>
<tr>
  <th>intervention_class</th>             <td>   -0.1029</td> <td>    0.046</td> <td>   -2.239</td> <td> 0.025</td> <td>   -0.193</td> <td>   -0.013</td>
</tr>
<tr>
  <th>post_period</th>                    <td>    0.0986</td> <td>    0.057</td> <td>    1.741</td> <td> 0.082</td> <td>   -0.012</td> <td>    0.210</td>
</tr>
<tr>
  <th>intervention_class:post_period</th> <td>    0.0517</td> <td>    0.065</td> <td>    0.795</td> <td> 0.426</td> <td>   -0.076</td> <td>    0.179</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
fit.get_robustcov_results(
    cov_type="cluster", groups=test_scores["class_id"]
).summary(slim=True)
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>score</td>  <th>  R-squared:         </th> <td>   0.006</td> 
</tr>
<tr>
  <th>Model:</th>              <td>OLS</td>   <th>  Adj. R-squared:    </th> <td>   0.005</td> 
</tr>
<tr>
  <th>No. Observations:</th> <td>  5034</td>  <th>  F-statistic:       </th> <td>1.057e+04</td>
</tr>
<tr>
  <th>Covariance Type:</th>  <td>cluster</td> <th>  Prob (F-statistic):</th> <td>6.46e-124</td>
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>    0.0911</td> <td>    0.039</td> <td>    2.342</td> <td> 0.021</td> <td>    0.014</td> <td>    0.168</td>
</tr>
<tr>
  <th>intervention_class</th>             <td>   -0.1029</td> <td>    0.048</td> <td>   -2.131</td> <td> 0.036</td> <td>   -0.199</td> <td>   -0.007</td>
</tr>
<tr>
  <th>post_period</th>                    <td>    0.0986</td> <td>    0.001</td> <td>   82.473</td> <td> 0.000</td> <td>    0.096</td> <td>    0.101</td>
</tr>
<tr>
  <th>intervention_class:post_period</th> <td>    0.0517</td> <td>    0.002</td> <td>   33.734</td> <td> 0.000</td> <td>    0.049</td> <td>    0.055</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are robust to cluster correlation (cluster)



As universal bootstrap is based on universal sampling, it is straightforward to implement complex designs. In this case, the intervention is applied at the class level, so we hash the class ID.


```python
edu_hashes = mmhash(test_scores["class_id"].values)
edu_null = np.zeros(bootstrap_replications)

for i, random in enumerate(
    np.vstack([test_randomization, control_randomization]).T
):
    test_random, control_random = np.multiply.outer(
        edu_hashes, random
    ).T
    test_weights = poisson_sample_weight(
        test_random, test_percent_class
    )
    control_weights = poisson_sample_weight(
        control_random, 1 - test_percent_class
    )

    df = (
        pd.concat(
            [
                test_scores.assign(test=1, weight=test_weights),
                test_scores.assign(
                    test=0, weight=control_weights
                ),
            ]
        )
        .query("weight > 0")
        .eval("test_post = test * post_period")
    )

    ols = LinearRegression().fit(
        df[["test", "post_period", "test_post"]], df["score"]
    )
    edu_null[i] = ols.coef_[-1]

edu_point_est = fit.params["intervention_class:post_period"]

fig, ax = plt.subplots()
ax.hist(edu_null)
ax.axvline(edu_point_est, color="red", ls="--")
ax.set_title(
    "Clustered universal bootstrap: null distribution vs. point"
    " estimate"
);
```


    
![png](/notebooks/universal-bootstrap_files/universal-bootstrap_30_0.png)
    


## Wrapping up

Universal bootstrap is simply the application of Poisson bootstrap leveraging the strengths of universal sampling. It provides true determinism; efficiency in dataset or streams of any size; and straightforward application of complex sampling designs. In my next post, I'll cover how this can be done in SQL.
