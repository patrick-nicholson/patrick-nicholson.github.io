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
CREATE
    OR REPLACE FUNCTION poisson_draws(lambda DOUBLE PRECISION)
    RETURNS poisson_draw[]
AS
$$
WITH
    RECURSIVE
    quantiles AS (
        SELECT
            0            AS x,
            EXP(-lambda) AS p,
            EXP(-lambda) AS s
        UNION ALL
        SELECT
            x + 1,
            p * lambda / (x + 1),
            s + p * lambda / (x + 1)
        FROM
            quantiles
        WHERE
            s < .999999
    ),
    rights AS (
        SELECT
            FLOOR(s * 4294967295 - 2147483648) AS upper,
            x                                  AS value
        FROM
            quantiles
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
        SELECT
            upper,
            value
        FROM
            rights
        UNION ALL
        SELECT
            maxval AS upper,
            value
        FROM
            intmax
    ),
    formatted AS (
        SELECT
            COALESCE(LAG(upper) OVER (ORDER BY value) + 1,
                     -2147483648) AS __poisson_lower,
            upper                 AS __poisson_upper,
            value                 AS __poisson_value
        FROM
            uppers
    )

SELECT
    ARRAY_AGG(ROW (__poisson_lower, __poisson_upper, __poisson_value)::poisson_draw)
FROM
    formatted
    ;
$$
    LANGUAGE sql;