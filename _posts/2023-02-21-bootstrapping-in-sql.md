---
layout: post
author: "Patrick Nicholson"
title: "Bootstrapping in SQL; or, getting s--- done with the tools you have"
excerpt: "SQL is the lingua franca of data platforms. It is many things, but a statistical language is not one of them. In this post, I'll cover an approach to bootstrap when it's your best, or only, option."
image:
    path: /notebooks/palpatine-bootstrap.jpg
---

![jpg](/notebooks/palpatine-bootstrap.jpg)

Bootstrapping has so many useful properties that I often want to apply it everywhere. In my [previous post](/2023/02/20/universal-bootstrap/), I covered the universal bootstrap, a method that combines universal hash functions and the Poisson bootstrap. While I feel that it has strong benefits even in local-analysis tools like Python, it certainly is not the required way to do bootstrapping, so readers may not have immediately grasped its generalization.

Many common tools built specifically for data science/engineering often have language APIs (e.g., Spark and Flink). In such cases, the bootstrapping methods discussed in that post can be relatively easily adapted to those APIs for an idiomatic and efficient approach. However, not all tools are like this (looking at you Snowflake), and we're not always asked or able to do analyses in them anyway.

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
    return Markdown(f"```sql\n{text}\n```")


iris.rename_axis("row_id").to_sql(
    "iris", postgres, if_exists="replace"
)
test_scores.to_sql(
    "test_scores", postgres, index=False, if_exists="replace"
)
```

## Baby wants SQL

Suppose `iris` is a Postgres table and we want to get the bootstrap distribution of a statistic. A naive way to do this is to repeatedly query the table. A better way is to use the universal bootstrap.

Postgres does not have a fast, non-cryptographic hash function by default. There are some extensions available that provide them, and you could write a user-defined function to expose one from another language. However, to demonstrate that this is possible in pure SQL, I use MD5 as my hashing function. Since MD5 returns 128 bits, I actually split this up into four 32-bit hashes. You can certainly treat each hash column separately, but it's straightforward to turn them into row values into a single column that is much sipler to use.




```sql

CREATE TYPE hash AS
(
    __hash_index INT,
    __hash_value INT
);


/**
  Generate up to 4 32-bit hashes
  
  Parameters
  ----------
  val TEXT  
    The source column to hash
  hashes INT
    The number of hashes to keep (between 1 and 4)
  
  Returns
  -------
  hashes SET
    Up to 4 hashes per row
 */
CREATE FUNCTION generate_hashes(val TEXT, hashes INT)
    RETURNS SETOF hash
AS
$$

WITH

-- apply md5 as the hash function
md5_applied AS (
    SELECT MD5(val::TEXT) AS __md5
),

-- turn the md5 into a 128 bitarray
md5_bits AS (
    SELECT
        ('x' || __md5)::BIT(128) AS __bits
    FROM
        md5_applied
),

-- get four 32-bit segments as 32-bit integers
with_hash_columns AS (
    SELECT
        (__bits << 0)::BIT(32)::INT  AS __hash0,
        (__bits << 32)::BIT(32)::INT AS __hash1,
        (__bits << 64)::BIT(32)::INT AS __hash2,
        (__bits << 64)::BIT(32)::INT AS __hash3
    FROM
        md5_bits
),

-- flatten the four columns into rows

-- intermediate step to index the hashes [0, 3]
-- (nested arrays wouldn't work)
rows_ugly AS (
    SELECT
        UNNEST(ARRAY [
            __hash0::BIGINT << 2,
            (__hash1::BIGINT << 2) | 1,
            (__hash2::BIGINT << 2) | 2,
            (__hash3::BIGINT << 2) | 3
            ]) AS __hash_with_index_bits
    FROM
        with_hash_columns
)

-- back to the correct format
SELECT
    (__hash_with_index_bits & 3)::INT  AS __hash_index,
    (__hash_with_index_bits >> 2)::INT AS __hash_value
FROM
    rows_ugly
WHERE
    (__hash_with_index_bits & 3) < hashes
    ;
$$ LANGUAGE sql;

```



Next, we need a way to randomize hashes. The approach is the same as in the previous post: multiply the uniform hash value by another random uniform integer. 
* Our source of random values in Postgres is random uniform values in (0, 1); these are easily converted to integers in $\left[ -2^{31}, 2^{31} \right)$
* Multiplication is not as simple because Postgres doesn't wrap integer math; it's easy enough to get around this by multiplying in `bigint` and modulo-ing back down to `int`




```sql

CREATE TYPE random_integer AS
(
    __random_index INT,
    __random_value INT
);

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
CREATE FUNCTION random_integers(num INT, seed DOUBLE PRECISION)
    RETURNS random_integer[]
AS
$$
SELECT SETSEED(seed);
WITH
    random AS (
        SELECT
            __random_index,
            FLOOR(RANDOM() * 4294967295 - 2147483648)::INT 
                AS __random_value
        FROM
            GENERATE_SERIES(0, num - 1) __random_index
    )

SELECT
    ARRAY_AGG(ROW (__random_index, __random_value)::random_integer)
FROM
    random
;
$$ LANGUAGE sql;


/**
  Combines hash generation with randomization

  Parameters
  ----------
  val TEXT
    The source column to hash
  hashes INT
    The number of hashes to keep (between 1 and 4)
  random_integers ARRAY
    Randomizing integers from random_integers function

  Returns
  -------
  hashes SET
    Up to `hashes` hashes per row
 */
CREATE FUNCTION generate_randomized_hashes(val TEXT, hashes INT, 
    random_integers random_integer[])
    RETURNS SETOF hash
AS
$$
SELECT
    ((rix).__random_index << 2) | __hash_index AS __hash_index,
    ((((__hash_value::BIGINT * (rix).__random_value) % 4294967295) 
        + 4294967295) % 4294967295)::BIT(32)::INT AS __hash_value
FROM
    generate_hashes(val, hashes) AS h,
    ( select unnest(random_integers)::random_integer as rix ) r
WHERE
    ((rix).__random_index << 2) | __hash_index < hashes
;
$$ LANGUAGE sql;

```



Next, we need to map uniform integers to Poisson values. This is straightforward using a [simple inverse transform sampling method](https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation) for a very high value of the CDF.




```sql

CREATE TYPE poisson_draw AS
(
    __poisson_lower INT,
    __poisson_upper INT,
    __poisson_value INT
);


/**
  Draw integer ranges for a Poisson distribution
  
  Parameters
  ----------
  lambda DOUBLE PRECISION
    The mean of the distribution
  
  Returns
  -------
  ARRAY
    Array of inclusive integer ranges for 99.9999% of the 
    distribution
 */
CREATE FUNCTION poisson_draws(lambda DOUBLE PRECISION)
    RETURNS poisson_draw[]
AS
$$
WITH
    RECURSIVE
    quantiles AS (
        SELECT 0 AS x, EXP(-lambda) AS p, EXP(-lambda) AS s
        UNION ALL
        SELECT x + 1, p * lambda / (x + 1), s + p * lambda / (x + 1) 
        FROM quantiles 
        WHERE s < .999999
    ),
    rights AS (
        SELECT 
            FLOOR(s * 4294967295 - 2147483648) AS upper, 
            x AS value
        FROM quantiles
    ),
    intmax AS (
        SELECT
            2147483647     AS maxval,
            MAX(upper)     AS maxright,
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
    ARRAY_AGG(ROW (lower, upper, value)::poisson_draw)
FROM
    lowers
;
$$ LANGUAGE sql;

```



Finally, we compose this all together. A bootstrap is configured by the sample fraction(s) we need to draw from the original dataset and the number of times we need to do it. For full reproducibility, we also expose a Postgres random number generator seed \[in Postgres, this is a float in (0, 1)\].




```sql

CREATE TYPE group_configuration AS
(
    __group_index   INT,
    __poisson_lower INT,
    __poisson_upper INT,
    __poisson_value INT
);

CREATE TYPE bootstrap_configuration AS
(
    __num_hashes      INT,
    __num_groups      INT,
    __groups          group_configuration[],
    __random_integers random_integer[]
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
    replications INT, seed DOUBLE PRECISION)
    RETURNS bootstrap_configuration
AS
$$

WITH
    num_groups AS (
        SELECT ARRAY_LENGTH(group_fractions, 1) AS num_groups
    ),
    draws AS (
        SELECT
            ix - 1 AS __group_index,
            UNNEST(poisson_draws(group_fractions[ix]))::poisson_draw 
                AS draw
        FROM
            GENERATE_SUBSCRIPTS(group_fractions, 1) ix
    ),
    groups AS (
        SELECT
            ARRAY_AGG(ROW(
                __group_index,
                (draw).__poisson_lower,
                (draw).__poisson_upper,
                (draw).__poisson_value
                )::group_configuration) AS __groups
        FROM
            draws
        WHERE
            0 < (draw).__poisson_value
    ),
    random_size AS (
        SELECT
            CEIL(replications::FLOAT * num_groups / 4)::INT 
                AS num_random
        FROM
            num_groups
    )

SELECT
    ROW (replications * num_groups, num_groups, __groups, 
         random_integers(num_random, seed))
FROM
    num_groups,
    groups,
    random_size
;
$$ LANGUAGE sql;


/**
  Configure a bootstrap with only one group

  Parameters
  ----------
  fraction DOUBLE PRECISION
    Resampling fraction
  replications INT
    The number of bootstrap replications
  seed DOUBLE PRECISION
    The random number generator seed
 */
CREATE FUNCTION configure_bootstrap(fraction DOUBLE PRECISION, 
    replications INT, seed DOUBLE PRECISION)
    RETURNS bootstrap_configuration
AS
$$
SELECT configure_bootstrap(ARRAY[fraction], replications, seed);
$$ LANGUAGE sql;


/**
  Resample a table based on the hash value of the input and a
  bootstrapping configuration

  Parameters
  ----------
  val TEXT  
    The source column to hash
  config bootstrap_configuration
    A configuration returned by the `configure_bootstrap` function
  
  Returns
  -------
  SET
    A set of rows for the bootstrap results
 */
CREATE FUNCTION poisson_bootstrap(val TEXT, 
    config bootstrap_configuration)
    RETURNS TABLE
            (
                __replication_index INT,
                __group_index       INT
            )
AS
$$
SELECT
    __hash_index / (config).__num_groups,
    (grp).__group_index
FROM
    generate_randomized_hashes(val, (config).__num_hashes, 
        (config).__random_integers),
    ( SELECT UNNEST((config.__groups))::group_configuration AS grp ) 
        groups, 
    LATERAL GENERATE_SERIES(1, (grp).__poisson_value)
WHERE
    __hash_index % (config).__num_groups = (grp).__group_index
    AND __hash_value BETWEEN (grp).__poisson_lower 
        AND (grp).__poisson_upper
;
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
        configure_bootstrap(1, 200, .5))
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


    
![png](/notebooks/bootstrapping-in-sql_files/bootstrapping-in-sql_14_0.png)
    


## Hypothesis testing

Yes, you can even do this in SQL. Since difference-in-differences has a simple analytical solution, I'm going to repeat the educational intervention analysis from my previous post. All of the functionality required to do this is included in the functions above. The only change I need to make in configuring a bootstrap for two groups (test and control).



```sql

with 

configure as (
    select configure_bootstrap(array[0.75, 0.25], 200, .5) as config
),

bootstrap as (
    select 
        __replication_index, 
        case when __group_index = 0 then 1 else 0 end test, 
        post_period as post,
        avg(score) as cell,
        count(1) as samples,
        count(distinct class_id) as units
    from
        test_scores,
        configure,
        lateral poisson_bootstrap(class_id::text, configure.config)
    group by
        1, 2, 3
),

pivot as (
    select
        __replication_index,
        sum(samples) as samples,
        sum(units * test) as test_units,
        sum(units * (1 - test)) as control_units,
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
    test_units,
    control_units,
    (test_post - test_pre) - (control_post - control_pre) as estimate
from
    pivot

```




```python
fig, ax = plt.subplots(nrows=2, sharex=True)

ax[0].hist(edu["estimate"])
ax[0].axvline(edu_point_est, color="red", ls="--")
ax[0].set_title(
    "SQL bootstrap clustered null distribution vs. point"
    " estimate"
)

ax[1].hist(edu_null)
ax[1].set_title(
    "Reference Poisson bootstrap null distribution from previous"
    " post"
)

fig.tight_layout();
```


    
![png](/notebooks/bootstrapping-in-sql_files/bootstrapping-in-sql_17_0.png)
    


## Wrapping up

You now have the power to bootstrap in SQL, unlocking richer analysis in a traditional RDBMS, visualization tools, Snowflake (sigh), or anywhere else you prefer to use SQL. Go make your DBadmins feel it.
