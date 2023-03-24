---
layout: post
author: "Patrick Nicholson"
title: "Bootstrapping in SQL; or, getting s--- done with the tools you have"
excerpt: "SQL is the lingua franca of data platforms. It is many things, but a statistical language is not one of them. In this post, I'll cover an approach to bootstrap when it's your best, or only, option."
notebook:
    path: /public/bootstrapping-in-sql.ipynb
image:
    path: /public/palpatine-bootstrap.jpg
---

![jpg](/public/palpatine-bootstrap.jpg)

Bootstrapping has so many useful properties that I often want to apply it everywhere. In my [previous post](/2023/02/20/universal-bootstrap/), I covered the universal bootstrap, a method that combines universal hash functions and the Poisson bootstrap. While I feel that it has strong benefits even in local-analysis tools like Python, it certainly is not the required way to do bootstrapping, so readers may not have immediately grasped its generalization.

Many common tools built specifically for data science/engineering often have language APIs (e.g., Spark and Flink). In such cases, the bootstrapping methods discussed in that post can be relatively easily adapted to those APIs for an idiomatic and efficient approach. However, not all tools are like this, and we're not always asked or able to do analyses in them anyway.

SQL is the _lingua franca_ of data platforms. It is many things, but a statistical language is not one of them. In this post, I'll cover an approach to bootstrap when it's your best, or only, option. However, to my knowledge there will be differences in implementation details or idioms that will need to differ between platforms. I'm only covering Postgres here, but it should be enough to get you started with any vendor.


```python
from sqlalchemy import create_engine

postgres = create_engine(
    "postgresql+psycopg://postgres:postgres"
    "@host.docker.internal:5432/postgres"
)


def read_postgres(query):
    """Simple wrapper to read from postgres"""
    from sqlalchemy import text

    with postgres.connect() as con:
        return pd.read_sql(text(query), con)


def execute_postgres(stmt):
    """Simple wrapper to execute a statement"""
    from sqlalchemy import text

    with postgres.connect() as con:
        con.execute(text(f"BEGIN;{stmt};END;"))


def display_sql(text):
    """Prettier display of my SQL strings"""
    return Markdown(f"```sql\n{text.strip()}\n```")


iris.rename_axis("row_id").to_sql(
    "iris", postgres, if_exists="replace"
)
test_scores.to_sql(
    "test_scores", postgres, index=False, if_exists="replace"
)
```

## Baby wants SQL

Suppose `iris` is a Postgres table and we want to get the bootstrap distribution of a statistic. A naive way to do this is to repeatedly query the table. A better way is to use the universal bootstrap.

Postgres does not have a fast, non-cryptographic hash function by default. There are some extensions available that provide them, and you could write a user-defined function to expose one from another language. However, to demonstrate that this is possible in pure SQL, I use MD5 as my hashing function. Postgres's MD5 returns 128 hex-encoded bits from a text input; I just take the first 32 bits as an integer (`int4` in Postgres).




```sql
/**
  Create a 32-bit hash from text

  Parameters
  ----------
  col TEXT
    The source column to hash

  Returns
  -------
  int4
    32-bit hash value
 */
CREATE OR REPLACE FUNCTION hash(col TEXT)
    RETURNS INT
AS
$$
SELECT ('x' || MD5(col))::BIT(128)::BIT(32)::INT
$$ LANGUAGE sql;
```



Next, we need a way to randomize hashes. The approach is the same as in the previous post: multiply the uniform hash value by another random uniform integer. 
* Our source of random values in Postgres is random uniform values in (0, 1); these are easily converted to integers in $\left[ -2^{31}, 2^{31} \right)$
* Multiplication is not as simple because Postgres doesn't wrap integer math; it's easy enough to get around this by multiplying in `bigint` and modulo-ing back down to `int`




```sql
/**
  Generate an arbitrary number of random integers (int4)

  Parameters
  ----------
  num INT
    The number of random values
  seed DOUBLE PRECISION
    The random number generator seed

  Returns
  -------
  ARRAY
    An array of length num
 */
CREATE OR REPLACE FUNCTION generate_random_integers(num INT)
    RETURNS INT[]
AS
$$
SELECT
    ARRAY_AGG(FLOOR(RANDOM() * 4294967296 + -2147483648)::INT)
FROM
    GENERATE_SERIES(1, num)
;
$$ LANGUAGE sql;


/**
  Performs wrapped 32-bit multiplication
  
  Parameters
  ----------
  x, y INT
    Values to multiply

  Returns
  -------
  INT
    Result of multiplication with wrapping
 */
CREATE OR REPLACE FUNCTION wrapped_multiply(x INT, y INT)
    RETURNS INT
AS
$$
SELECT ((((x::BIGINT * y) % 4294967296) + 4294967296) 
    % 4294967296)::BIT(32)::INT
$$ LANGUAGE SQL;
```



Next, we need to map uniform integers to Poisson values. This is straightforward using a [simple inverse transform sampling method](https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation) for a very high value of the CDF.




```sql
CREATE TYPE poisson_cdf_interval AS
(
    cdf_lower INT,
    cdf_upper INT,
    poisson_value INT
);


/**
  Map the 32-bit integer range to a Poisson distribution

  Parameters
  ----------
  lambda DOUBLE PRECISION
    The mean of the distribution

  Returns
  -------
  ARRAY
    Array of inclusive integer ranges and associated Poisson values
 */
CREATE OR REPLACE FUNCTION generate_poisson_cdf(
    lambda DOUBLE PRECISION)
    RETURNS poisson_cdf_interval[]
AS
$$
WITH RECURSIVE
    quantiles AS (
        SELECT 0 AS x, EXP(-lambda) AS p, EXP(-lambda) AS s
        UNION ALL
        SELECT x + 1, p * lambda / (x + 1), s + p * lambda / (x + 1)
        FROM quantiles
        WHERE s < .999999
    ),
    rights AS (
        SELECT 
            FLOOR(s * 4294967296 + -2147483648) AS upper, 
            x AS value
        FROM quantiles
    ),
    intmax AS (
        SELECT
            2147483647 AS maxval,
            MAX(upper) AS maxright,
            MAX(value) + 1 AS value
        FROM
            rights
        GROUP BY
            maxval
        HAVING
            MAX(upper) < 2147483647
    ),
    uppers AS (
        SELECT upper, value
        FROM rights
        UNION ALL
        SELECT maxval AS upper, value
        FROM intmax
    ),
    lowers AS (
        SELECT
            COALESCE(LAG(upper) OVER (ORDER BY value) + 1, 
                -2147483648) AS lower,
            upper,
            value
        FROM
            uppers
    )

SELECT
    ARRAY_AGG(ROW (lower, upper, value)::poisson_cdf_interval)
FROM
    lowers
$$ LANGUAGE sql;
```



Finally, we compose this all together. A bootstrap is configured by the sample fraction(s) we need to draw from the original dataset and the number of times we need to do it. For full reproducibility, we also expose a Postgres random number generator seed \[in Postgres, this is a float in (0, 1)\].




```sql
CREATE TYPE bootstrap_group_configuration AS
(
    group_index INT,
    random_integers INT[],
    poisson_cdf poisson_cdf_interval[]
);


/**
  Configure a bootstrap

  Parameters
  ----------
  group_fractions ARRAY
    Resampling fractions for each group
  replications INT
    The number of bootstrap replications
  seed DOUBLE PRECISION
    The random number generator seed
 */
CREATE FUNCTION configure_bootstrap(
    group_fractions DOUBLE PRECISION[], 
    replications INT, 
    seed DOUBLE PRECISION)
    RETURNS bootstrap_group_configuration[]
AS
$$
SELECT SETSEED(seed);
SELECT
    ARRAY_AGG(ROW(groups.*)::bootstrap_group_configuration)
FROM (
    SELECT
        ix - 1,
        generate_random_integers(replications),
        generate_poisson_cdf(frac)
    FROM
        UNNEST(group_fractions) WITH ORDINALITY AS t(frac, ix)
) groups;
$$ LANGUAGE sql;


/**
  Resample a table based on the hash value of the input and a
  bootstrapping configuration

  Parameters
  ----------
  col TEXT
    The source column to hash
  config bootstrap_group_configuration[]
    A configuration array returned by the `configure_bootstrap`
    function

  Returns
  -------
  SET
    A set of rows for the bootstrap results
 */
CREATE OR REPLACE FUNCTION poisson_bootstrap(
    col TEXT, 
    config bootstrap_group_configuration[])
    RETURNS TABLE
            (
                __replication_index INT,
                __group_index       INT
            )
AS
$$
WITH weights as (
    SELECT
        randomize.hash_ix - 1 as replication,
        group_config.group_index,
        poisson.poisson_value
    FROM
        (select hash(col) as hashed) AS hashing,
        (select (unnest(config)::bootstrap_group_configuration).*) 
            AS group_config,
        lateral unnest(group_config.random_integers) 
            WITH ORDINALITY AS randomize(randint, hash_ix),
        lateral (select (unnest(group_config.poisson_cdf)
            ::poisson_cdf_interval).*) AS poisson
    WHERE
        wrapped_multiply(hashing.hashed, randomize.randint) 
            BETWEEN poisson.cdf_lower AND poisson.cdf_upper
)

SELECT
    replication,
    group_index
FROM
    weights,
    lateral generate_series(1, poisson_value)
$$ LANGUAGE sql;
```



The `poisson_bootstrap` is called as a `LATERAL` expression in a query, producing and average of (number of groups) $\times$ (number of replications) $\times$ (average of sample fractions) output rows for each input row, with `__replication_index` and `__group_index` columns providing the necessary identification.

Baby, let's bootstrap. It's simply an aggregation including the necessary bootstrap columns in the grouping.




```sql
select 
    __replication_index, 
    count(1) as samples, 
    count(distinct row_id) as units, 
    avg(sepal_length) as estimate
from 
    iris, 
    lateral poisson_bootstrap(row_id::text, 
        configure_bootstrap(ARRAY[1], 200, .5))
group by 1
```




```python
fig, ax = plt.subplots(
    nrows=4, figsize=(plt.rcParams["figure.figsize"][:1] * 2)
)

ax[1].sharex(ax[0])
ax[3].sharex(ax[2])

ax[0].hist(baby["samples"], bins=20)
ax[0].axvline(len(iris), color="red", ls="--")
ax[0].set_title(
    "Sample size in each replication vs. original size"
)

ax[1].hist(baby["units"], bins=20)
ax[1].set_title("Unique units in each replication")

ax[2].hist(baby["estimate"], bins=20)
ax[2].axvline(iris["sepal_length"].mean(), color="red", ls="--")
ax[2].set_title("SQL bootstrap distribution vs. point estimate")

ax[3].hist(poisson_baby_null, bins=20)
ax[3].set_title(
    "Reference Poisson bootstrap distribution from previous post"
)

fig.tight_layout();
```


    
![png](/public/bootstrapping-in-sql_files/nb_14_0.png)
    


## Hypothesis testing

Yes, you can even do this in SQL. Since difference-in-differences has a simple analytical solution, I'm going to repeat the educational intervention analysis from my previous post. All of the functionality required to do this is included in the functions above. The only change I need to make is configuring the bootstrap for two groups (test and control).




```sql
with 

configure as (
    select configure_bootstrap(array[0.75, 0.25], 200, .5) 
        as config
),

bootstrap as (
    select 
        __replication_index, 
        case when __group_index = 0 then 1 else 0 end test, 
        post_period as post,
        avg(score) as cell,
        count(1) as samples
    from
        test_scores,
        configure,
        lateral poisson_bootstrap(class_id::text, 
            configure.config)
    group by
        1, 2, 3
),

pivot as (
    select
        __replication_index,
        sum(samples) as samples,
        sum(cell * test * post) as test_post,
        sum(cell * test * (1 - post)) as test_pre,
        sum(cell * (1 - test) * post) as control_post,
        sum(cell * (1 - test) * (1 - post)) as control_pre
    from
        bootstrap
    group by
        __replication_index
)

select
    __replication_index,
    samples,
    (test_post - test_pre) - (control_post - control_pre) 
        as estimate
from
    pivot
```




```python
fig, ax = plt.subplots(nrows=2, sharex=True)

bins = np.histogram_bin_edges(np.r_[edu_point_est, edu_null], 15)

ax[0].hist(edu["estimate"], bins=bins)
ax[0].axvline(edu_point_est, color="red", ls="--")
ax[0].set_title(
    "SQL bootstrap clustered null distribution vs. point"
    " estimate"
)

ax[1].hist(edu_null, bins=bins)
ax[1].set_title(
    "Reference Poisson bootstrap null distribution from previous"
    " post"
)

fig.tight_layout();
```


    
![png](/public/bootstrapping-in-sql_files/nb_17_0.png)
    


## Wrapping up

You now have the power to bootstrap in SQL, unlocking richer analysis in a traditional RDBMS, visualization tools, Snowflake (sigh), or anywhere else you prefer to use SQL. Go make your DBadmins feel it.
