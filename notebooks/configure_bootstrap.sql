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
CREATE OR REPLACE FUNCTION configure_bootstrap(group_fractions DOUBLE PRECISION[],
                                               replications INT,
                                               seed DOUBLE PRECISION)
    RETURNS bootstrap_configuration
AS
$$

WITH
    num_groups AS (
        SELECT ARRAY_LENGTH(group_fractions, 1) AS num_groups
    ),
    draws AS (
        SELECT
            ix - 1                                                   AS __group_index,
            UNNEST(poisson_draws(group_fractions[ix]))::poisson_draw AS draw
        FROM
            GENERATE_SUBSCRIPTS(group_fractions, 1) ix
    ),
    groups AS (
        SELECT
            ARRAY_AGG(ROW (__group_index,
                (draw).__poisson_lower,
                (draw).__poisson_upper,
                (draw).__poisson_value)::group_configuration) AS __groups
        FROM
            draws
        WHERE
            0 < (draw).__poisson_value
    ),
    random_size AS (
        SELECT
            CEIL(replications::FLOAT * num_groups / 4)::INT AS num_random
        FROM
            num_groups
    )

SELECT
    ROW (replications * num_groups, num_groups, __groups, random_integers(num_random, seed))
FROM
    num_groups,
    groups,
    random_size a
$$ LANGUAGE sql;
