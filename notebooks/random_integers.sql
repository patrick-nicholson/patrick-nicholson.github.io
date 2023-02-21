CREATE TYPE random_integer AS
(
    __random_index INT,
    __random_value INT
);

CREATE FUNCTION random_integers(num INT, seed DOUBLE PRECISION)
    RETURNS random_integer[]
AS
$$
SELECT SETSEED(seed);
WITH
    random AS (
        SELECT
            __random_index,
            FLOOR(RANDOM() * 4294967295 - 2147483648)::INT AS __random_value
        FROM
            GENERATE_SERIES(0, num - 1) __random_index
    )

SELECT
    ARRAY_AGG(ROW (__random_index, __random_value)::random_integer)
FROM
    random
;
$$ LANGUAGE sql;
