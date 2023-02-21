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

CREATE OR REPLACE FUNCTION generate_randomized_hashes(val TEXT, hashes INT, random_integers random_integer[])
    RETURNS SETOF hash
AS
$$
SELECT
    ((rix).__random_index << 2) | __hash_index AS __hash_index,
    ((((__hash_value::BIGINT * (rix).__random_value) %
       4294967295) + 4294967295)
        % 4294967295)::BIT(32)::INT    AS __hash_value
FROM
    generate_hashes(val, hashes) AS h,
    (select unnest(random_integers)::random_integer as rix) r
WHERE
    ((rix).__random_index << 2) | __hash_index < hashes
    ;
$$ LANGUAGE sql;
