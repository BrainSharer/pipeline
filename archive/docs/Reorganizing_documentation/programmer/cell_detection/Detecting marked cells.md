### Evaluation of cell detector



### scores

Each detection has 30 scores, we compute the mean and the std of the scores.

The mean/std score is used as follows:

if the std is larger than $a \approx 6$  we label this as confidently not a cell (Would be worthwhile seeing what these detections are)

if the std is smaller than $a$ then

* if the mean is smaller than $ -b \approx -1.5$: confident no
* If the mean is larger than $b$: confident yes
* if the mean is between $-b$ and $b$ : unconfident detection.

$b$ is the mean of the std values.

#### A detection region

A detection region is a set of centroids such that the distance between any two detections is at most $r=10 \mu m$  . The radius $r$ is set to be twice the maximal diameter of a single marked cell.

In multi-threshold detections the detection region contains the centroids from all of the thresholds.

We also use detection regions to count both human labels and computer labels.

In almost all cases the detection region is unique. If it is not unique we take the union of all of the possibilities.

#### Choosing a threshold

Different thresholds can create different centroids for the same detection region. The detections used are the ones that correspond to the treshold such that the mean of the detection scores is highest.

for groups with mixed sure and unsure detections, the sure detection with highest mean score is selected

#### Counts

We evaluate the performance by counting the number of machine centroids and human centroids.

* Human centroids are labeled as "positive" or "negative"
* Machine centroids are labeled as "confident" or "not confident"

* Each detection region is labeled by 4 numbers:
  * $p$ for number of human labeled positives
  * $n$ for number of human labeled negatives
  * $c$ for number of confident detections
  * $u$ for number of non-confident detections.
* Some common configurations:
  * $c=1,u=0, n=0,p=0$  Unverified detection
  * $c=1,u=0,n=0,p=1$ Verified detection
  * $c=0,u=1,p=1,n=0$ unconfident identified  as positive
  * $c=0,u=1,p=0,n=0$ unconfident identified as negative
  * $c=2,u=0,p=2,n=2$ a pair of detections at wrong locations.
  * There are many other configurations, many of which will appear only very few times.
  * As we scan the detection regions, we label each one of them with the tuple $(c,u,p,n)$. Later on we count the different types and produce the errors table.



 