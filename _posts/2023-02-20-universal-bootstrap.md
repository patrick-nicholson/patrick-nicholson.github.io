---
layout: post
author: patrick-nicholson
title: "Universal bootstrap: a superpower"
excerpt: "Bootstrapping is a commonly used in computational statistics and machine learning for uncertainty quantification, hypothesis testing, and ensembling. By combining universal sampling with Poisson bootstrap, I show how the universal bootstrap unlocks incredibly sophisticated analysis at any scale in any tool."
---

> _Look on my computational methods, ye theorists, and despair_

[Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) is a commonly used technique for computational statistics and machine learning. 
* [Uncertainty quantification](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)): an approximate distribution of a sample statistic (e.g., mean) is the empirical distribution of the same statistic calculated over bootstrap samples
* [Bootstrap tests](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Bootstrap_hypothesis_testing): a distribution-free hypothesis test by using the empirical distribution of the test statistic from boostrap samples that randomize the test design
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
from sklearn.utils.murmurhash import murmurhash3_32 as mmhash
from sqlalchemy import create_engine
from uuid import uuid4

%matplotlib inline

random_state = np.random.RandomState(12345)
postgres = create_engine(
    "postgresql+psycopg://postgres:postgres@host.docker.internal:5432/postgres"
)


def read_postgres(query):
    """Simple wrapper to read from postgres"""
    from sqlalchemy import text

    with postgres.connect() as con:
        return pd.read_sql(text(query), con)


def compose_read_postgres(statement, *cte, **format_kwargs):
    """Helper to compose SQL fragments and run the query"""
    cte = ", ".join(cte)
    if cte:
        cte = f"with {cte}"
    query = (f"{cte} {statement}").format(**format_kwargs)
    return read_postgres(query)


def display_sql(text):
    """Prettier display of my SQL strings"""
    return Markdown(f"```sql\n{text}\n```")
```

## Key insight

Bootstrapping is based on sampling with replacement. [Poisson sampling](https://en.wikipedia.org/wiki/Poisson_sampling) is one such method. For a sample size $N$ and resampling size $S$, the resampling weights for each observation follows the $\text{Pois} \left( \frac{S}{N} \right)$ distribution. The [Poisson bootstrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Poisson_bootstrap), commonly used for bootstrapping streams and large datasets, is the particular case where $N = S$, i.e., $\text{Pois}(1)$.

In [my earlier post](https://patrick-nicholson.github.io/2023/02/13/universal-sampling/), I demonstrated a method of sampling from a Poisson distribution based on universal hash functions. Composing this with Poisson sampling gives us the universal bootstrap.


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


def sql_poisson_weights(column, edges, lower=None, upper=None):
    lower = lower or 0
    upper = upper or len(edges)
    whens = "\n".join(
        f"\twhen {column} < {e} then {i + lower}"
        for i, e in enumerate(edges)
    )
    return f"case\n{whens}\n\telse {upper}\nend"
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




    5.906666666666667




```python
n = len(iris)

bs_estimates = (
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
ax.hist(bs_estimates)
ax.axvline(iris["sepal_length"].mean(), c="red")
ax.set_title("Bootstrap distribution of a sample statistic");
```


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_11_0.png)
    


A single universal bootstrap estimate is the sample statistic weighted by Poisson weights. The universal bootstrap distribution repeats this by randomizing the hashes and repeating the weighting process.


```python
hashes = mmhash(iris.index.values.astype(np.int32))
weights = poisson_sample_weight(hashes, 1)
np.average(iris["sepal_length"], weights=weights)
```




    5.938410596026489




```python
randomization = random_state.randint(
    -(2**31), 2**31, bootstrap_replications, dtype=np.int32
)
weights = poisson_sample_weight(
    np.multiply.outer(hashes, randomization), 1
)
poisson_bs_estimates = (
    weights * iris["sepal_length"].values[:, None]
).sum(axis=0) / weights.sum(axis=0)

fig, ax = plt.subplots()
ax.hist(poisson_bs_estimates)
ax.axvline(iris["sepal_length"].mean(), c="red")
ax.set_title(
    "Universal bootstrap distribution of a sample statistic"
);
```


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_14_0.png)
    


Of course, you can also do this by repeating observations based on the Poisson weights.


```python
values = np.tile(iris["sepal_length"], bootstrap_replications)
weights = weights.T.ravel()
replications = np.repeat(
    np.arange(bootstrap_replications), len(iris)
)
values, weights, replications = (
    values[weights > 0],
    weights[weights > 0],
    replications[weights > 0],
)

repeated_estimates = (
    pd.Series(np.repeat(values, weights))
    .groupby(np.repeat(replications, weights))
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


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_16_0.png)
    


## Baby wants SQL

Once a little data scientist learns the bootstrap, they want to do it everywhere. They quickly learn that things are not so simple.

Suppose `iris` is a Postgres table and we want to get the bootstrap distribution of a statistic. A naive way to do this is to repeatedly query the table. A better way is to use the universal bootstrap.

Postgres does not have a fast, non-cryptographic hash function by default. There are some extensions available that provide them and you could write a user-defined function, but one must suffer to use SQL, so I am going to use MD5 as my hashing function. Since MD5 returns 128 bits, I am actually going to split this up into four 32-bit hashes.




```sql


-- apply md5 as the hash function
__md5_applied as (
    select 
        -- packs all input columns into a single column. i find 
        -- this more convenient that trying select all but a few 
        -- columns
        row({table}.*)::{table} as __data, 
        md5({column}::text) as __md5 
    from 
        {table}
),

-- turn the md5 into a 128 bitarray
__md5_bits as (
    select 
        __data, 
        ('x' || __md5)::bit(128) as __bits 
    from 
        __md5_applied
),

-- get four 32-bit segments as 32-bit integers
with_hash_columns as (
    select
        __data,
        (__bits << 0 )::bit(32)::int as __hash0,
        (__bits << 32)::bit(32)::int as __hash1,
        (__bits << 64)::bit(32)::int as __hash2,
        (__bits << 64)::bit(32)::int as __hash3
    from
        __md5_bits
)

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
      <th>__hash0</th>
      <th>row_id</th>
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
      <td>-808640380</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-993377736</td>
      <td>1</td>
      <td>0</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-937528691</td>
      <td>2</td>
      <td>0</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-322189186</td>
      <td>3</td>
      <td>0</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1468008839</td>
      <td>4</td>
      <td>0</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



You can certainly treat each hash column separately, but it's (somewhat) straightforward to turn them into row values in a column.




```sql


-- flatten the four columns into rows

-- intermediate step to index the hashes [0, 3]
-- (nested arrays wouldn't work)
__rows_ugly as (
    select
        __data,
        unnest(array[
            __hash0::bigint << 2, 
            (__hash1::bigint << 2) | 1, 
            (__hash2::bigint << 2) | 2, 
            (__hash3::bigint << 2) | 3
        ]) as __hash_with_index_bits
    from
        with_hash_columns
),

-- split into hash and index
with_hash_rows as (
    select 
        __data, 
        (__hash_with_index_bits & 3)::int as __hash_index, 
        (__hash_with_index_bits >> 2)::int as __hash_value
    from
        __rows_ugly
)

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
      <th>__hash_index</th>
      <th>__hash_value</th>
      <th>row_id</th>
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
      <td>-808640380</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1781176849</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1726472185</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1726472185</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>-993377736</td>
      <td>1</td>
      <td>0</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



Randomization is similar to what I showed in Python, but requires handling integer overflow manually. I generate random values and put them in the query as a values expression.




```sql


-- random values
randomization as (
    select 
        x.*
    from (
        values {values} 
    ) as x(__random_index, __random_value)
),

-- randomize
-- math doesn't wrap in postgres, so
-- 1. convert to long
-- 2. multiply by random number
-- 3. positive modulo in the uint32 range
-- 4. convert back to signed int32
with_randomization as (
    select
        __data,
        (__random_index << 2) | __hash_index as __hash_index,
        ((((__hash_value::bigint * __random_value) % 4294967295)
            + 4294967295) % 4294967295)::bit(32)::int 
            as __hash_value
    from
        with_hash_rows, 
        randomization
)

```




```python
# only need 1/4 of randomization values because i have 4 hash functions
random_values_expr = ", ".join(
    map(str, enumerate(randomization[::4]))
)
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
      <th>__hash_index</th>
      <th>__hash_value</th>
      <th>row_id</th>
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
      <td>1528266260</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>-935289221</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>-2465396</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12</td>
      <td>-863430936</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>-1926202386</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



Weights are added with a simple case statement on the hash value. The example below shows this for a $\text{Pois}(1)$.




```sql

with_poisson_weights as (
    select
        __data,
        __hash_index,
        {poisson_case} as __weight
    from
        with_randomization
    where 
        {poisson_case} > 0
)

```






```sql
case
	when __hash_value < -567453479 then 0
	when __hash_value < 1012576689 then 1
	when __hash_value < 1802591773 then 2
	when __hash_value < 2065930135 then 3
	when __hash_value < 2131764725 then 4
	when __hash_value < 2144931643 then 5
	when __hash_value < 2147126129 then 6
	when __hash_value < 2147439628 then 7
	when __hash_value < 2147478815 then 8
	when __hash_value < 2147483647 then 9
	else 10
end
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
      <th>__hash_index</th>
      <th>__weight</th>
      <th>row_id</th>
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
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



At this point, we have the data we need to estimate the bootstrap distribution.


```python
_ = """
    select 
        sum((__data).sepal_length * __weight) / sum(__weight) 
            as estimate
    from 
        with_poisson_weights 
    group by
        __hash_index
"""

display_sql(_)
```




```sql

    select 
        sum((__data).sepal_length * __weight) / sum(__weight) 
            as estimate
    from 
        with_poisson_weights 
    group by
        __hash_index

```




```python
estimates = compose_read_postgres(
    _,
    hash_cte,
    flatten_cte,
    randomize_cte,
    poisson_cte,
    table="iris",
    column="row_id",
    values=random_values_expr,
    poisson_case=poisson_case,
)

fig, ax = plt.subplots()
ax.hist(estimates["estimate"])
ax.set_title("Postgres universal bootstrap distribution");
```


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_34_0.png)
    


Like before, we can also expand the rows based on weight if a weighted statistic is unavailable or overly complex.




```sql

    select 
        avg((__data).sepal_length) as estimate
    from 
        with_poisson_weights,
        lateral (select * from generate_series(1, __weight)) t
    group by
        __hash_index

```



## Universal bootstrap for hypothesis testing

While baby's first bootstrap was fun and all, the practical power of bootstrapping is really revealed when estimating the $H_0$ distribution of a hypothesis test. 

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

The universal boostrap is changed only slightly: we now generate test and control weights according to their sample sizes. Each replication has pseudo-test and pseudo-control samples, from which we calculate the test value.

The test is then comparing the null distribution to the observed statistic from the true test and control groups.


```python
test_randomization, control_randomization = random_state.randint(
    -(2**31), 2**31, (2, bootstrap_replications), np.int32
)

hashes = (
    purchases["customer_uuid"]
    .map(mmhash)
    .values.astype(np.int32)
)

test_weights = poisson_sample_weight(
    np.multiply.outer(hashes, test_randomization),
    test_percent_offer,
)
test = (test_weights.T * purchases["purchases"].values).sum(
    axis=1
) / test_weights.sum(axis=0)
del test_weights

control_weights = poisson_sample_weight(
    np.multiply.outer(hashes, control_randomization),
    1.0 - test_percent_offer,
)
control = (
    control_weights.T * purchases["purchases"].values
).sum(axis=1) / control_weights.sum(axis=0)
del control_weights

null_distribution = test - control

control, test = purchases.groupby("received_offer")[
    "purchases"
].mean()
actual = test - control

fig, ax = plt.subplots()
ax.hist(null_distribution)
ax.axvline(actual, color="red", ls="--")
ax.set_title(
    "Universal boostrap test: null distribution vs. observed"
    " difference"
);
```


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_40_0.png)
    


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
# - post_period: score is for the second test (post intervention, 
#                if received)
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
  <th>Dep. Variable:</th>      <td>score</td>   <th>  R-squared:         </th> <td>   0.005</td>
</tr>
<tr>
  <th>Model:</th>               <td>OLS</td>    <th>  Adj. R-squared:    </th> <td>   0.005</td>
</tr>
<tr>
  <th>No. Observations:</th>  <td>  4924</td>   <th>  F-statistic:       </th> <td>   8.939</td>
</tr>
<tr>
  <th>Covariance Type:</th>  <td>nonrobust</td> <th>  Prob (F-statistic):</th> <td>6.65e-06</td>
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>   -0.0086</td> <td>    0.041</td> <td>   -0.208</td> <td> 0.835</td> <td>   -0.090</td> <td>    0.073</td>
</tr>
<tr>
  <th>intervention_class</th>             <td>    0.0203</td> <td>    0.047</td> <td>    0.429</td> <td> 0.668</td> <td>   -0.073</td> <td>    0.113</td>
</tr>
<tr>
  <th>post_period</th>                    <td>    0.0976</td> <td>    0.059</td> <td>    1.664</td> <td> 0.096</td> <td>   -0.017</td> <td>    0.213</td>
</tr>
<tr>
  <th>intervention_class:post_period</th> <td>    0.0540</td> <td>    0.067</td> <td>    0.806</td> <td> 0.420</td> <td>   -0.077</td> <td>    0.185</td>
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
  <th>Dep. Variable:</th>     <td>score</td>  <th>  R-squared:         </th> <td>   0.005</td> 
</tr>
<tr>
  <th>Model:</th>              <td>OLS</td>   <th>  Adj. R-squared:    </th> <td>   0.005</td> 
</tr>
<tr>
  <th>No. Observations:</th> <td>  4924</td>  <th>  F-statistic:       </th> <td>1.333e+04</td>
</tr>
<tr>
  <th>Covariance Type:</th>  <td>cluster</td> <th>  Prob (F-statistic):</th> <td>6.95e-129</td>
</tr>
</table>
<table class="simpletable">
<tr>
                 <td></td>                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                      <td>   -0.0086</td> <td>    0.048</td> <td>   -0.179</td> <td> 0.859</td> <td>   -0.105</td> <td>    0.087</td>
</tr>
<tr>
  <th>intervention_class</th>             <td>    0.0203</td> <td>    0.055</td> <td>    0.366</td> <td> 0.715</td> <td>   -0.090</td> <td>    0.130</td>
</tr>
<tr>
  <th>post_period</th>                    <td>    0.0976</td> <td>    0.001</td> <td>   68.663</td> <td> 0.000</td> <td>    0.095</td> <td>    0.100</td>
</tr>
<tr>
  <th>intervention_class:post_period</th> <td>    0.0540</td> <td>    0.002</td> <td>   33.014</td> <td> 0.000</td> <td>    0.051</td> <td>    0.057</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors are robust to cluster correlation (cluster)



As universal bootstrap is based on universal sampling, it is  straightforward to implement complex designs. In this case, the intervention is applied at the class level, so we hash the class ID.


```python
hashes = mmhash(test_scores["class_id"].astype(np.int32).values)
estimates = np.zeros(bootstrap_replications)

for i, random in enumerate(
    np.vstack([test_randomization, control_randomization]).T
):
    test_random, control_random = np.multiply.outer(
        hashes, random
    ).T
    test_weights = poisson_sample_weight(
        test_random, test_percent_class
    )
    control_weights = poisson_sample_weight(
        control_random, 1 - test_percent_class
    )

    df = pd.concat(
        [
            test_scores.assign(
                test=1, weight=test_weights
            ).query("weight > 0"),
            test_scores.assign(
                test=0, weight=control_weights
            ).query("weight > 0"),
        ]
    ).eval("test_post = test * post_period")

    ols = LinearRegression().fit(
        df[["test", "post_period", "test_post"]], df["score"]
    )
    estimates[i] = ols.coef_[-1]

fig, ax = plt.subplots()
ax.hist(estimates)
ax.axvline(
    fit.params["intervention_class:post_period"],
    color="red",
    ls="--",
)
ax.set_title(
    "Clustered universal boostrap: null distribution vs. point"
    " estimate"
);
```


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_47_0.png)
    


## Must... test... SQL

Yes, you can even do this in SQL. Since difference-in-differences has a simple analytical solution, we can even test the educational intervention. The necessary update is changing the randomization we used before to create a pseudo-test and pseudo-control groups and apply different weights to each. 


```python
test_scores.to_sql(
    "test_scores", postgres, index=False, if_exists="replace"
)

test_control_random_values = ", ".join(
    map(
        str,
        enumerate(
            np.r_[test_randomization, control_randomization]
        ),
    )
)

*_, test_edges = poisson_thresholds(test_percent_class)
*_, control_edges = poisson_thresholds(1 - test_percent_class)
test_case = sql_poisson_weights("__hash_value", test_edges)
control_case = sql_poisson_weights("__hash_value", control_edges)
```




```sql


with_test_design as (
    select
        __data,
        -- collapse the hashes from num_hashes to (num_hashes / 2)...
        __hash_index / 2 as __hash_index,
        -- ...where one of the collapsed hashes is now for 
        -- the test group
        __hash_index % 2 as __test,
        case 
            when (__hash_index % 2) = 0 then {control_case} 
            else {test_case} 
            end as __weight
    from
        with_randomization
)

```


```sql

, __aggregation as (
    select
        __hash_index,
        __test,
        (__data).post_period,
        sum((__data).score * __weight) / sum(__weight) as cell
    from
        with_test_design
    where
        __weight > 0
    group by
        __hash_index,
        __test,
        post_period
),

__pivot as (
    select
        __hash_index,
        sum(cell * __test * post_period) as test_post,
        sum(cell * __test * (1 - post_period)) as test_pre,
        sum(cell * (1 - __test) * post_period) as control_post,
        sum(cell * (1 - __test) * (1 - post_period)) as control_pre
    from
        __aggregation
    group by
        __hash_index
)

select
    __hash_index,
    (test_post - test_pre) - (control_post - control_pre) as estimate
from
    __pivot

```




```python
postgres_bootstrap_test = compose_read_postgres(
    _,
    hash_cte,
    flatten_cte,
    randomize_cte,
    test_design_cte,
    table="test_scores",
    column="class_id",
    values=test_control_random_values,
    test_case=test_case,
    control_case=control_case,
)
```


```python
fig, ax = plt.subplots()
ax.hist(postgres_bootstrap_test["estimate"])
ax.set_title(
    "Null distribution from Postgres universal bootstrap"
);
```


    
![png](/notebooks/sampling/universal-bootstrap_files/universal-bootstrap_53_0.png)
    


## Wrapping up

Universal bootstrap is simply the application of Poisson bootstrap leveraging the strengths of universal sampling. It provides true determinism; efficiency in dataset or streams of any size; and straightforward application of complex sampling designs.  
