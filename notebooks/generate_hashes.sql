CREATE type hash as (
                __hash_index INT,
                __hash_value INT
            );


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

SELECT
    (__hash_with_index_bits & 3)::INT  AS __hash_index,
    (__hash_with_index_bits >> 2)::INT AS __hash_value
FROM
    rows_ugly
WHERE
    (__hash_with_index_bits & 3) < hashes
    ;
$$ LANGUAGE sql;