---
layout: post
author: patrick-nicholson
title: "Universal sampling: better sampling for a better tomorrow"
excerpt: "Universal hash functions efficiently and deterministically map arbitrary input to uniformly distributed integers. In this post, I demonstrate how to leverage these functions for sampling from datasets and distributions."
---

[Universal hashing](https://en.wikipedia.org/wiki/Universal_hashing) is a powerful primitive for statistical analysis at scale. Universal hash functions efficiently and deterministically map inputs to integers that are uniformly distributed within the binary range of an integer type (e.g., a 64-bit long).

Why would you want to do this? Deterministic sampling allows for complete reproducibility within and between instances of 1 to $\infty$. This allows for completely memoryless and stateless sampling from a dataset or draws from a distribution. 
* You can cluster sample in a single pass
* You can sample the same identities in a stream or between different batch runs
* You can efficiently bootstrap and apply computational statistics at scale
* You can build approximate representations of sets or datasets ("sketches")
* ...and a lot of other things I'm not thinking about right now but you'll eventually discover

The two universal families that I come across most often for data science, data engineering, and machine learning applications are [MurmurHash](https://github.com/aappleby/smhasher) and [xxHash](https://github.com/Cyan4973/xxHash):
* `scikit-learn` uses MurmurHash for [feature hashing via the hashing trick](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing)
* The Scala standard library includes a [MurmurHash implementation](https://www.scala-lang.org/api/2.12.x/scala/util/hashing/MurmurHash3$.html)
* Apache Spark includes functions for both [MurmurHash](https://spark.apache.org/docs/latest/api/sql/index.html#hash) and [xxHash](https://spark.apache.org/docs/latest/api/sql/index.html#xxhash64). Spark has long used MurmurHash for [hash partitioning](https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-HashPartitioning.html) to efficiently and uniformly distributed data for computation.

Universal hashing is by no means limited to these families; MurmurHash and xxHash are particularly relevant to DS/DE/ML work because they are very fast and do not try to meet cryptographic requirements. And there are other hash families in this vein. For example, Snowflake includes an unspecified [64-bit uniform hash function](https://docs.snowflake.com/en/sql-reference/functions/hash.html).


```python
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from pyspark.sql import SparkSession
from scipy import stats
from sklearn.utils.murmurhash import murmurhash3_32 as mmhash
from typing import Union
from uuid import uuid4

random_state = np.random.RandomState(12345)
spark = SparkSession.builder.getOrCreate()

%matplotlib inline
```

## Hashing

Let's take a look at universal hashing in Python with `scikit-learn`'s MurmurHash implementation. For exmaple inputs, I create integer range and random normal arrays and hash their values. Despite the differences in the input distributions, each value in each array is unique so the hash values of both are approximately uniformly distributed in the 32-bit signed range $\left[ -2^{31}, 2^{31} \right)$. 


```python
n = 10**4

integers = np.arange(n, dtype=np.int32)
normals = random_state.normal(size=n)

assert len(set(integers)) == n
assert len(set(normals)) == n

integer_hashes = mmhash(integers)
normal_hashes = np.array([mmhash(bytes(_)) for _ in normals])

assert len(set(integer_hashes)) == n
assert len(set(normal_hashes)) == n
assert all(integer_hashes != normal_hashes)

fig, ax = plt.subplots(ncols=2, nrows=2)

for i, x in enumerate(
    [integers, integer_hashes, normals, normal_hashes]
):
    ax[i // 2, i % 2].hist(x)

ax[0, 0].set_title("Values")
ax[0, 1].set_title("Hashes")
ax[0, 0].set_ylabel("Integer range")
ax[1, 0].set_ylabel("Random normal")

fig.tight_layout();
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_3_0.png)
    


These hashes are uniformly distributed in this range because each bit is randomly activated.


```python
bytes_righty = (
    integer_hashes.astype("uint32").view("uint8").reshape(-1, 4)
)

# double check that i have the ops right
assert all(
    integer_hashes.astype("uint32")
    == np.bitwise_or.reduce(
        np.left_shift(bytes_righty, np.arange(4) * 8).T
    )
)

bytes_lefty = bytes_righty[:, ::-1]
bits = np.unpackbits(bytes_lefty).reshape(-1, 32)
```


```python
bits.mean()
```




    0.500484375




```python
fig, ax = plt.subplots()
ax.hist(bits.sum(axis=1))
ax.set_title("Active bits per hash");
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_7_0.png)
    


If you're in an environment without these fast non-cryptographic hash functions, remember that cryptographic functions provide you many random bytes. For example, we can take the last 4 bytes of MD5 hashes (8 hex characters).


```python
md5_randint = np.array(
    [
        int(hashlib.md5(bytes(value)).hexdigest()[-8:], 16)
        for value in integers
    ],
    dtype=np.uint32,
).astype(np.int32)
```


```python
fig, ax = plt.subplots()
ax.hist(md5_randint)
ax.set_title("Distribution of truncated MD5 hashes");
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_10_0.png)
    


## Random sampling

<div class="alert alert-block alert-info" > I'm going to compare the <code>pandas</code> sampling interface with universal sampling. The <code>pandas</code> interface is generally representative of the tools I'm familiar with (PySpark, R base/<code>dplyr</code>, SQL, etc.). It would not surprise me to learn that there are richer implementations around that don't have some of the downsides I'll point out below. </div> 


The simplest application of universal hashing is random sampling. We start with a representative dataframe:
* `user_id` is a user UUID
* `activity_id` is an activity UUID
* `score` is a purely random sample from a normal distribution


```python
users = 10**4
avg_posts_per_user = 5

df = pd.DataFrame(
    [
        (user_id, uuid4(), score)
        for posts in random_state.poisson(
            avg_posts_per_user, users
        )
        for user_id in (uuid4(),)
        for score in random_state.normal(0, 1, posts)
    ],
    columns=["user_id", "activity_id", "score"],
)
```

Dataframes in various packages or platforms usually make _simple_ random sampling very easy. 


```python
sample_rate = 0.15
df.sample(frac=sample_rate)
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
      <th>user_id</th>
      <th>activity_id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42123</th>
      <td>2146d00d-da8c-4b7a-afe2-069d0122e654</td>
      <td>b2d392b5-34c8-45ed-a10d-12647630c824</td>
      <td>0.613257</td>
    </tr>
    <tr>
      <th>15501</th>
      <td>9a30100c-bb59-4537-ac06-407104c278a5</td>
      <td>9e8cb4d1-ca02-48b2-ac8a-4cdc5781545f</td>
      <td>0.489327</td>
    </tr>
    <tr>
      <th>36599</th>
      <td>5c7df545-ee6b-4404-8ca1-acf2456437bb</td>
      <td>90b7512f-b055-420c-a390-3d973e70a2e5</td>
      <td>-0.224887</td>
    </tr>
    <tr>
      <th>29152</th>
      <td>6c006f1f-f585-435e-88ac-e4216fcb941f</td>
      <td>7f622e7e-082b-41bd-b3c2-0e717dc4da71</td>
      <td>-0.597781</td>
    </tr>
    <tr>
      <th>48048</th>
      <td>c4b5892a-0766-4d9a-8cb0-629697c7748e</td>
      <td>bebcc679-66cf-49be-a362-69601525fd9c</td>
      <td>0.722863</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22002</th>
      <td>62f0f65e-ba69-4e80-b7f0-84005e3696c6</td>
      <td>8c3c54e4-8c44-4120-82a9-27f2a8c90fea</td>
      <td>-0.449790</td>
    </tr>
    <tr>
      <th>3576</th>
      <td>a9a0d17a-87e4-4db2-b266-a9e52224cf35</td>
      <td>ca765b75-d088-4b92-8a9b-91c5b10b7f64</td>
      <td>-0.600585</td>
    </tr>
    <tr>
      <th>14450</th>
      <td>73b70191-0e65-4455-95cf-ef5681076b9c</td>
      <td>48783b4b-c8b1-4f98-9579-96c1cc75d994</td>
      <td>1.635014</td>
    </tr>
    <tr>
      <th>10454</th>
      <td>9bdf3218-931d-435c-8e95-b00115fefedc</td>
      <td>0ba40f6a-f577-4955-b756-bf39f99051c1</td>
      <td>-0.115104</td>
    </tr>
    <tr>
      <th>47444</th>
      <td>388fb874-f67d-42ba-a7cc-6d66bace4993</td>
      <td>41447a12-e3b0-44be-8cde-829f7a466889</td>
      <td>0.295622</td>
    </tr>
  </tbody>
</table>
<p>7465 rows × 3 columns</p>
</div>



Reproducibility is controlled through seeds for random numbers. Given identical inputs, this method provides reproducible sample by row.


```python
sample0 = df.sample(frac=sample_rate, random_state=2323)
sample1 = df.sample(frac=sample_rate, random_state=2323)
sample0.equals(sample1)
```




    True



More general forms of random sampling are usually incovenient. For example, sampling half of our users requires deduplicating the users and sampling from that set, then filtering the original data.


```python
(
    df[["user_id"]]
    .drop_duplicates()
    .sample(frac=sample_rate)
    .merge(df)
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
      <th>user_id</th>
      <th>activity_id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2949f3e1-0541-4f47-8008-3a927fa572c5</td>
      <td>cdc62abd-2fa7-4720-84db-bc10622d8e21</td>
      <td>-0.771325</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2949f3e1-0541-4f47-8008-3a927fa572c5</td>
      <td>6a8cb546-ce66-43b0-b5e5-dad4b3f0916b</td>
      <td>1.041870</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2949f3e1-0541-4f47-8008-3a927fa572c5</td>
      <td>68ea7ee1-120f-4ad9-ae3f-1716fce8f7e8</td>
      <td>-1.711934</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2949f3e1-0541-4f47-8008-3a927fa572c5</td>
      <td>e71a7993-8483-49fb-b915-ee2d08889b16</td>
      <td>0.624260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2949f3e1-0541-4f47-8008-3a927fa572c5</td>
      <td>9b4665df-c187-4f5e-8fc2-8274697cee39</td>
      <td>0.337053</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7446</th>
      <td>8dcf2c48-a709-4355-8656-46968f073e57</td>
      <td>634d97f9-6cad-4633-90dd-f0f07d21429f</td>
      <td>0.881056</td>
    </tr>
    <tr>
      <th>7447</th>
      <td>bf0d2973-edb4-4abf-8712-f82ea2cba6f5</td>
      <td>a9e49605-5d35-47fb-a56f-37a0f88375ea</td>
      <td>-2.327481</td>
    </tr>
    <tr>
      <th>7448</th>
      <td>bf0d2973-edb4-4abf-8712-f82ea2cba6f5</td>
      <td>b3c24ccc-987b-4638-a7c4-258f8c0eb865</td>
      <td>1.913991</td>
    </tr>
    <tr>
      <th>7449</th>
      <td>bf0d2973-edb4-4abf-8712-f82ea2cba6f5</td>
      <td>d8cc684b-f9aa-41e8-ab80-069ad3cf67a6</td>
      <td>-1.274627</td>
    </tr>
    <tr>
      <th>7450</th>
      <td>bf0d2973-edb4-4abf-8712-f82ea2cba6f5</td>
      <td>7ccb6baf-9b9e-48e7-b591-89774c7aedee</td>
      <td>0.223246</td>
    </tr>
  </tbody>
</table>
<p>7451 rows × 3 columns</p>
</div>



And there is no answer to sampling from non-identical inputs: these dataframe methods are based on random numbers drawn independently of the data.


```python
sample0 = df.sample(frac=sample_rate, random_state=2323)
sample1 = df.iloc[::-1].sample(
    frac=sample_rate, random_state=2323
)
sample0.equals(sample1.iloc[::-1])
```




    False




```python
users = df[["user_id"]].drop_duplicates(ignore_index=True)
sample0 = users.iloc[:1000].sample(
    frac=sample_rate, random_state=2323
)
sample1 = users.iloc[500:1500].sample(
    frac=sample_rate, random_state=2323
)

l = sample0.merge(users.iloc[500:1000])
r = sample1.merge(users.iloc[500:1000])
l.merge(r, how="outer", indicator=True).groupby("_merge").count()
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
      <th>user_id</th>
    </tr>
    <tr>
      <th>_merge</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>left_only</th>
      <td>69</td>
    </tr>
    <tr>
      <th>right_only</th>
      <td>67</td>
    </tr>
    <tr>
      <th>both</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



The kinds of problems can be addressed with universal simple random sampling. For a hash function $f_{[a,b)}$ producing hash values $h_i \in \left[a, b \right)$ and sample rate $r$, it's simplest and most efficient to transform the rate to a ceiling value such that we keep $h_i < \left( a + r * (b - a) \right)$.

We can still do independent sampling based on a value that's unique to a row (here just the row number).


```python
sample_ceiling = -(2**31) + int(sample_rate * 2**32)
df[df.index.map(mmhash) < sample_ceiling]
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
      <th>user_id</th>
      <th>activity_id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>8ff6d097-7e7d-429b-9182-bf5966cdc605</td>
      <td>2b0583d2-9493-4308-8440-a090ff21faf7</td>
      <td>-0.204278</td>
    </tr>
    <tr>
      <th>20</th>
      <td>bccdc4c9-e0be-40c8-aee9-bb5db1eb7073</td>
      <td>45244d60-a7d3-43bd-83b1-56fffe0afa2d</td>
      <td>-1.006406</td>
    </tr>
    <tr>
      <th>33</th>
      <td>e394c3b2-1888-4dbe-91a3-1987f4277f79</td>
      <td>3560e6b0-638c-4f28-ab80-07cc2af4cb15</td>
      <td>1.430888</td>
    </tr>
    <tr>
      <th>34</th>
      <td>e394c3b2-1888-4dbe-91a3-1987f4277f79</td>
      <td>5cc82930-6965-44e0-b12b-155c91db989a</td>
      <td>0.155472</td>
    </tr>
    <tr>
      <th>35</th>
      <td>e394c3b2-1888-4dbe-91a3-1987f4277f79</td>
      <td>80d12d35-5b0a-4bcf-9c1d-bf5beca646d1</td>
      <td>0.463298</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49718</th>
      <td>89d94a90-41ce-465a-8690-f303042709f6</td>
      <td>05397ddf-fc37-4502-a933-bd266ae87828</td>
      <td>0.067397</td>
    </tr>
    <tr>
      <th>49720</th>
      <td>89d94a90-41ce-465a-8690-f303042709f6</td>
      <td>7653091c-26f2-43c3-9dd1-0194923234b1</td>
      <td>0.658050</td>
    </tr>
    <tr>
      <th>49727</th>
      <td>a7fc8cba-66b9-4b3f-b1ad-3dd3aa566c4b</td>
      <td>dda4b2ea-c37e-4cec-a83f-ca4e147fafa0</td>
      <td>-1.502790</td>
    </tr>
    <tr>
      <th>49744</th>
      <td>e9d84d83-bdde-4edc-b2a7-09867a5e08b9</td>
      <td>e7d0e95b-a54e-4b9e-8355-c4fbd50edc98</td>
      <td>-1.757438</td>
    </tr>
    <tr>
      <th>49754</th>
      <td>cda6db16-26fe-48b1-8f45-732614895062</td>
      <td>edd696fb-834e-4462-b68a-4bd2c21f1678</td>
      <td>-0.753189</td>
    </tr>
  </tbody>
</table>
<p>7413 rows × 3 columns</p>
</div>



Sampling all data for a subset of users is as simple as changing the input to the hash function.


```python
df[df["user_id"].astype(np.bytes_).map(mmhash) < sample_ceiling]
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
      <th>user_id</th>
      <th>activity_id</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>0d196f13-8acd-4c1f-820e-6c16d717311b</td>
      <td>23afcefe-d18f-467c-ab60-c0d92c3a2514</td>
      <td>1.379005</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0d196f13-8acd-4c1f-820e-6c16d717311b</td>
      <td>f72bbd4e-ad9e-4855-be28-bc91e02d558d</td>
      <td>0.228165</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0d196f13-8acd-4c1f-820e-6c16d717311b</td>
      <td>c01a9f08-10ee-4ca9-a626-7b3371619426</td>
      <td>-0.969587</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0d196f13-8acd-4c1f-820e-6c16d717311b</td>
      <td>2e199bac-e3ff-4e86-9e7e-1075b5b9e31c</td>
      <td>0.158903</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1e05d5e5-b264-42ac-87dd-34170bdce3aa</td>
      <td>0f16d9f2-749f-4ead-a139-5cb694a94a85</td>
      <td>-1.613581</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49724</th>
      <td>a7fc8cba-66b9-4b3f-b1ad-3dd3aa566c4b</td>
      <td>0f191644-490f-4a2b-bc2a-f873808ace9b</td>
      <td>-0.737764</td>
    </tr>
    <tr>
      <th>49725</th>
      <td>a7fc8cba-66b9-4b3f-b1ad-3dd3aa566c4b</td>
      <td>8e54d914-59d7-4d1a-bc94-523dc1178041</td>
      <td>-1.022173</td>
    </tr>
    <tr>
      <th>49726</th>
      <td>a7fc8cba-66b9-4b3f-b1ad-3dd3aa566c4b</td>
      <td>a072ee32-3e53-4d0d-bc85-933e29f3ff17</td>
      <td>0.229212</td>
    </tr>
    <tr>
      <th>49727</th>
      <td>a7fc8cba-66b9-4b3f-b1ad-3dd3aa566c4b</td>
      <td>dda4b2ea-c37e-4cec-a83f-ca4e147fafa0</td>
      <td>-1.502790</td>
    </tr>
    <tr>
      <th>49728</th>
      <td>a7fc8cba-66b9-4b3f-b1ad-3dd3aa566c4b</td>
      <td>0422aae9-e2a6-46ce-9f24-4dfd118d35a5</td>
      <td>1.788407</td>
    </tr>
  </tbody>
</table>
<p>7489 rows × 3 columns</p>
</div>



The properties of universal hash functions allow us to sample from non-identical data, such as a stream or distributed dataset. For example, we can sample the same users from differing subsets.


```python
users = df[["user_id"]].drop_duplicates(ignore_index=True)
batch0 = users.iloc[:1000]
batch1 = users.iloc[500:1500]
overlap = users.iloc[500:1000]

l = batch0[
    batch0["user_id"].astype(np.bytes_).map(mmhash)
    < sample_ceiling
]
r = batch1[
    batch1["user_id"].astype(np.bytes_).map(mmhash)
    < sample_ceiling
]
l.merge(r, how="outer", indicator=True).merge(overlap).groupby(
    "_merge"
).count()
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
      <th>user_id</th>
    </tr>
    <tr>
      <th>_merge</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>left_only</th>
      <td>0</td>
    </tr>
    <tr>
      <th>right_only</th>
      <td>0</td>
    </tr>
    <tr>
      <th>both</th>
      <td>79</td>
    </tr>
  </tbody>
</table>
</div>



### Multiple hash functions

A single hash function deterministically maps an input. Sometimes we want a different hash function. For example, we don't want to sample the same set of users for every analysis. This is particularly relevant for universal sampling from distributions because pseudo-random number generation algorithms often require multiple seed inputs.

The easiest way to change a hash function is to salt the input. For example, any operation that produces a new, unique value not equal to the input will yield a new, uncorrelated hash. Such an operation includes the hash function itself.


```python
np.corrcoef(mmhash(integers), mmhash(integers + 1))[0, 1]
```




    0.01906422861062026




```python
np.corrcoef(mmhash(integers), mmhash(mmhash(integers)))[0, 1]
```




    -0.006641027550636124



For hash functions that accept multiple inputs, the salt can just be another argument. This is an equivalent strategy to the above: we're essentially converting the original input to bytes (if it wasn't already) and appending salting bytes.


```python
import pyspark.sql.functions as F

F.hash(F.col("column"), F.lit("salt"))
```




    Column<'hash(column, salt)'>



Repeated hashing is computationally expensive, can be error-prone (using an incorrect salting method), and can introduce serialized computation. This can outweigh its benefits (simplicity and strong guarantees) with large datasets or many hash functions. In practice, we can produce new pseudorandom numbers by integer multiplication with random numbers from the same discrete uniform distribution.


```python
def randomize_hashes(hashes, k, random_seed):
    """Randomize hash values k times"""
    hashes = hashes.astype(np.int32)
    random_state = np.random.RandomState(random_seed)
    random_integers = random_state.randint(
        -(2**31), 2**31, k, np.int32
    )
    return np.multiply.outer(hashes, random_integers)


np.corrcoef(
    np.vstack(
        [
            integer_hashes,
            randomize_hashes(
                integer_hashes, k=5, random_seed=858623
            ).T,
        ]
    )
)[0, 1:]
```




    array([ 0.0010438 ,  0.0122731 ,  0.00666873, -0.00060876,  0.01207721])



### Universal sampling from the uniform distribution

Suppose we want a value from a uniform distribution. Well, the universal hash functions describe above produce such a pseudo-random value.

Mapping the hash value to arbitrary discrete and continuous uniform distributions is trivial:
* A hash value $h$ can be mapped to an arbitrary discrete uniform $U \lbrace a, b \rbrace$ by $ \left[ h \mod (b - a) \right] + a $, where $\text{mod}$ is the positive modulo operator
* A hash function $f$ with bits $b$ produces a hash value $h$ that can be mapped to an arbitrary continuous uniform $U \left[a, b \right]$ by $ \left( \dfrac{h}{2^b} + \begin{cases} .5 & \text{if}\ f\ \text{is signed} \\ 0  & \text{if}\ f\ \text{is unsigned} \end{cases} \right) \cdot (b - a) + a $

### Universal sampling from the normal distribution

[There are several ways to generate values from the normal distribution that leverage samples from the uniform distribution](https://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution). According to the [Irwin-Hall distribution](https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Approximating_a_Normal_distribution), the scaled, centered sum of 12 random uniform values is approximately normally distributed. This is trivial to implement and is accurate enough for most applied work.


```python
n = 10**6

# distinct hashes for those distinct values
hashes = np.array(
    [mmhash(bytes(_), positive=True) for _ in normals],
    dtype=np.int32,
)

# approximate a normal from the hashes
irwin_hall_approx = (
    randomize_hashes(hashes, k=12, random_seed=73529) / 2**32
).sum(axis=1)


# visualize

fig, ax = plt.subplots(2, 2, sharex=True)

ax[0, 0].set_title("Random normal")
ax[0, 0].hist(normals, bins=30)

ax[1, 1].set_title("Irwin-Hall approximation")
ax[1, 1].hist(irwin_hall_approx, bins=30)

ax[0, 0].sharex(ax[1, 0])
ax[0, 0].sharey(ax[1, 1])

ax[1, 0].set_title("Joint distribution")
ax[1, 0].hist2d(
    normals, irwin_hall_approx, bins=30, cmap="inferno"
)

fig.delaxes(ax[0, 1])
plt.tight_layout();
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_37_0.png)
    


As long as an evaluation uses the same random integers for permutation, this opens the benefits of universal hashing to random normal sampling: completely independent and deterministic sampling for every distinct input to the hash function.

More exact samples can be generated by converting the hashes to continuous uniform samples (discussed above), including the non-iterative [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform).

More approximate samples can be generated with inverse transform sampling, which we used below for Poisson sampling.

### Universal sampling from the Poisson distribution

The Poisson distribution is well-suited for [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) when [the value of $\lambda$ (the mean of the distribution) is relatively small](https://en.wikipedia.org/wiki/Poisson_distribution#Random_variate_generation). Given the most common uses of the random the Poisson use such small $\lambda$ values, inverse transformation can be our default method for Poisson universal sampling.

The inverse transform method simply relates values of the cumulative distribution function to values of the distribution. In discrete cases, starting at zero this can be simplified to a single array where the index represents the value from the distribution. 

For index-based inverse transforms, we can simply evaluate inverse CDF within our tolerated range. For our small values of $\lambda$, in the Poisson distribution, the lower bound of this range will be zero. Index edges are inversely mapping the CDF values along this range to the range of our hash values.


```python
def poisson_sampling(hashes, lam, tol=1e-12):
    pois = stats.poisson(lam)
    lower, upper = np.ceil(pois.isf([1 - tol, tol]))

    edges = (
        pois.cdf(np.arange(lower, upper)) * 2**32 - 2**31
    ).astype(np.int32)

    poisson = pois.rvs(
        size=len(hashes), random_state=random_state
    )
    univseral_poisson = (
        np.searchsorted(edges, hashes, side="right") + lower
    )

    fig, ax = plt.subplots()
    num_bins = int(
        min(max(poisson.max(), univseral_poisson.max()) - 1, 30)
    )
    _, bins, _ = ax.hist(
        poisson, bins=num_bins, label="SciPy samples", alpha=0.5
    )
    ax.hist(
        univseral_poisson,
        bins=bins,
        label="Universal samples",
        alpha=0.5,
    )
    ax.set_title(f"Samples from the Pois({lam}) distribution")
    ax.legend()


poisson_sampling(integer_hashes, 2)
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_40_0.png)
    


This method is still perfectly valid with values of $\lambda$ much larger than typically used, but it may be computationally inefficient.


```python
poisson_sampling(integer_hashes, 10000)
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_42_0.png)
    


## Universal sampling from arbitrary distributions via approximate inverse transformation

Every distribution will have a best method for generating random values from universal hash seeds. However, in practice these won't always be tractable or even possible (e.g., infrastructure constraints).

Inverse transformation provides a general method of approximation in these cases. The approach is simple:
1. Evenly divide the (0, 1) interval 
2. Calculate the quantile at each value in the interval
3. Integerize the interval to the hash range 
4. Use the insertion position of hash values into the interval array to choose the quantile, breaking ties by choosing the quantile closest to the median

The precision of the approximation is controlled by the size of the lookup table we're willing to create. 


```python
def inverse_approximation(
    hashes, distribution, tol=1e-12, table_size=1000
):
    # cdf is full linear space in the tolerance range
    cdf = np.linspace(tol, 1 - tol, table_size)

    # values are evaluated at the CDF and padded with inf
    values = np.r_[-np.inf, distribution.isf(cdf[::-1]), np.inf]

    # edges are the integerized cdf
    # padded with the extrema to match size of values
    edges = np.r_[
        np.int32(-(2**31)),
        (cdf * 2**32 - 2**31).astype(np.int32),
        np.int32(2**31 - 1),
    ]

    # get index in edges for the hashes
    # choose the side closer to the center
    index = np.where(
        hashes < 0,
        np.searchsorted(edges, hashes, side="right"),
        np.searchsorted(edges, hashes),
    )

    samples = distribution.rvs(
        size=len(hashes), random_state=random_state
    )
    inverse_samples = values[index]

    fig, ax = plt.subplots()
    num_bins = 30
    if all(samples == np.floor(samples)):
        sample_range = max(
            samples.max() - samples.min(),
            inverse_samples.max() - inverse_samples.min(),
        )
        num_bins = min(30, int(sample_range))

    _, bins, _ = ax.hist(
        samples, bins=num_bins, label="SciPy samples", alpha=0.5
    )
    ax.hist(
        inverse_samples,
        bins=bins,
        label="Universal samples",
        alpha=0.5,
    )
    ax.legend()
    return ax
```


```python
ax = inverse_approximation(
    integer_hashes, stats.poisson(10**6)
)
ax.set_title("Approximate samples from Pois($10^6$)");
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_45_0.png)
    



```python
ax = inverse_approximation(integer_hashes, stats.norm(10, 10))
ax.set_title("Approximate samples from N(10, 10)");
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_46_0.png)
    



```python
ax = inverse_approximation(integer_hashes, stats.gamma(1, 2))
ax.set_title("Approximate samples from $\Gamma(1, 2)$");
```


    
![png](/notebooks/sampling/universal-sampling_files/universal-sampling_47_0.png)
    


### Wrapping up

The immediate applications of universal sampling from data are evident. It may be less clear why you might care about universal sampling from distributions. If you noticed that I didn't cover sampling with replacement, there is at least one very important application of sampling from distributions that I'll cover in my next post: Poisson resampling and bootstrap.
