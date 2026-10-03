# Week 1 Lab Submission — Know Your Data

AI Principles and Practice · COMP-0987 · Topic 01, Unit 02

One section below for each section of the lab. Answer as you go, rather than
at the end — most of these are the checkpoint questions you have already talked
through at the bench.

Short answers. One or two sentences unless it says otherwise. Delete each
`TODO` as you finish that section, then run `python make_submission.py`.

| | |
|---|---|
| Student number | 20068336 |
| Name | Joe Wemyss |
| Date | 2026-09-16 |

---

## Section 1 — First contact

**1.1 How many rows and columns, and how many literal blanks, does the file
have? Which three columns carry codes that are not in the documentation, and
how many rows does each affect?**

There are 30,000 rows and 25 columns. There are no literal blanks.

EDUCATION, MARRIAGE and PAY_0 contain codes that are undocumented. The breakdown is:

|**EDUCATION**|**MARRIAGE**|**PAY_0**|
|---|---|---|
|345|54|17496|

**1.2 Why is the `PAY_0` finding more serious than the `EDUCATION` one, even
though both are undocumented codes?**

The PAY_0 finding is more serious because of the scale. It far outstrips the amount of missing codes
for the other two

**1.3 The documented "paid duly" code (−1) defaults at 16.8%; the undocumented
−2 and 0 default at 13.2% and 12.8%. One sentence proposing what −2 and 0 most
likely mean, and a second on how confident you are and why behaviour is weaker
evidence than a codebook.**

This evidence seems to suggest that the -2 and 0 bear some correltion to paying on time or early. I would suggest that the -2 value means early and 0 means duly

My confidence in this is low. I would much prefer to have the codebook tell me excactly what the values mean. The fact that -2 correlates to a higer default rate makes the inference even more weak

---

## Section 2 — Summary statistics

**2.1 When do the mean and median disagree on this dataset, and which would you
report? Why is the IQR the safer measure of spread here?**

The mean and median really disagree on the BILL_AMT1 and PAY_AMT1. I would report on the median
since the reason for the disagreement is caused by a very large outlier in both cases (873552 for
PAY_AMT1 and 964511 for BILL_AMT1). Most people would not have paid anywhere near the mean.

This is also why the IQR is safer to measure the spread. Since its just the middle, dropping the
outliers, it gives a much more realistic view of the "middle" of the data here. The SD is more
affected by the huge outliers.

**2.2 `PAY_AMT1` has a skew of 14.7 and an excess kurtosis of 415. Say what
that means in plain words to someone who has never heard of kurtosis, and name
one modelling consequence.**

Both of these values are very high.

The skew is telling us that the tail is to the right. If it were
0, it would be balanced. If it were negative, it would be to the left. A positive 14 tells us it is
very far to the right. This aligns with the extremely high top value for PAY_AMT1

The kurtosis is a measure of how extreme it is. It doesn't care about left or right, it just tells
us that it is very far to whatever side it is skewed towards. Combined with the skew, we know that
the tail extends very far to the right. This again aligns with the very high value for PAY_AMT1

A modeling consequence of this is that we would need to be aware of these traits before modelling.
We now know that IQR is more useful than SD for this dataset, and median is more useful than mean.
These traits should be taken into account when modelling

**2.3 You redrew the `PAY_AMT1` histogram on a log scale. What does the
log-scaled version show that the raw one cannot?**

The raw version showed very little info. The reason for that was that it was essentially an empty
graph, with such a thin, tall bar on the left, and tiny histograms streaching off in the distance.

Even without a small constant, the log scale made it much easier to see that the vast majority of
the data was on the left. The small constant (I used 1) made it even easier to see how much data was
on the left

---

## Section 3 — Correlation

**3.1 One example where Pearson understates a relationship and one where it
overstates it. Give both coefficients.**

Pearson understates the relationship between the BILL_AMT1 and PAY_AMT1 Pearson said 0.14 to
Spearman 0.5

Pearson overstates the relationship between LIMIT_BAL and BILL_AMT1. Pearson said 0.29 to Spearman
0.05

**3.2 Adjacent bill months correlate at about 0.95. Name one problem that
causes for a linear model's coefficients, and one feature you could build from
the six columns that might carry more than any of them alone.**

The 0.95 correlation suggests that these features are very strongly correlated. This adds complexity
for linear coefficents, because we are dealing with 6 columns instead of 1. Perhaps the average
would be a better metric than the 6 individuals, or even whether the bill amount is growing or
shrinking.

**3.3 The strongest correlation with the target in the whole table is 0.325.
Write down a number for how accurate you expect next week's models to be. You
will check this against your Week 2 results.**

My expectation would be that the correlation would remain roughly the same, as I would expect the
current most recent data to decay in the same trend. I will say 0.325

---

## Section 4 — Processing

**4.1 Why did `df.isna().sum()` return zero on a dataset with hundreds of
unknown values?**

That function will only look for missing values. Just because a value is uncategorised does not mean
its "missing"

**4.2 Two or three sentences arguing against `df.dropna()` here. Defend the
specific claim: dropping those 399 rows would destroy information *and* bias
what remains.**

I think that the strongest argument against dropping those values is the fact that we can see that
they are not random. There is definitely a correlation between the default rate and the education.
That means that even if we do not know what those values mean, we should still use them to calculate
the potential default rate

**4.3 The IQR and z-score rules flagged 2,400 and 686 values. Why the gap, and
which do you believe? Then: 590 clients have a negative bill. One sentence on
what that means, and one on why deleting those rows would be a mistake.**

This is similar to how we chose the IQR over SD in the last section. Since our dataset is very
heavily right skewed, it means that SD is high, which drives down the z value. I would be inclined
to lean towards the IQR again here.

About the negative bill, that probably means the customer is in credit. They may be overpaying their
bill, or getting refunds etc. I would consider this a feature which should be analysed instead of
something to delete

---

## Section 5 — Augmentation

**5.1 Why must the split come before the resampling, and never the other way
round? What does `stratify=y` protect you from?**

We split first to stop the test set from leaking into the training data. If we did SMOTE first, the
synthetic rows would be synthesised from the rows we would be using for testing, which would result
in the model looking more accurate than it actually was. It would also mess up the test since the we
would have a balanced default rate in the test data.

The stratify=y is because of how imbalanced the set is. A pure random split could end up with no
defaults in the test set. The stratify=y maintains a balance on the field we are trying to predict

**5.2 SMOTE raised recall from 0.23 to 0.64, dropped precision from 0.74 to
0.36, and accuracy fell from 0.81 to 0.67. Two sentences on which setting you
would ship to a bank, and what you would need to know to decide.**

I would be more inclined to ship the model that includes SMOTE because of the big jump in recall.
That means we would catch more defaulters with it. This is based on the assumption that a default is
more costly than a false alarm. Knowing more about the business case would help with this decision

---

## Section 6 — Leakage-safe pipelines

**6.1 What single change turned an F1 of 0.665 into 0.473, and which of the two
is the truth?**

This is the issue from 5.1. We did SMOTE on the whole dataset, which means the model was trained on
the test data. That is why I would believe the lower 0.473 value from when SMOTE was only performed
on the training set

**6.2 Name the four preparation steps inside your `ColumnTransformer` and say
what each one learns from the training data. Why is a pipeline a structural fix
rather than a stylistic preference?**

* Numeric median imputation -- learns the median of the fields we flagged as numeric
* Numeric scaling -- learns a scale for values to ensure they are on a similar scale
* Categorical most frequent imputation -- learns what are the most frequent values from the fields
we flagged as Categorical
* One-hot encoding -- Learns the values of the categories so that they can be split out into features

The pipeline is a structural fix because it ensures that the data is not leaking by applying the
preparation steps only on the training data

**6.3 Does your pipeline run end to end? Give the accuracy you got, and say how
it compares with predicting "no default" for everyone.**

Yes, my pipeline ran end to end. I got 0.812, which is higher than the 0.779 for predicting a
default for everyone

---

## Section 7 — Governance and documentation

### 7.1 Data dictionary

All 25 columns. The Notes column should carry something you could only know by
having done Sections 1–4, not a restatement of the column name. Paste your
completed table here, or say which file in your submission holds it.

| Column | Type | Allowed values | Notes for a future user |
|---|---|---|---|
| `ID` | identifier | 1–30000 | Row identifier, not a feature. Drop before modelling. |
| `LIMIT_BAL` |continuous (NT$) |10,000 - 1,000,000 | right-skewed (skew 0.99)|
| `SEX` |binary category|1 = male, 2 = female| protected attibute (see datasheet)|
| `EDUCATION` | category|1 = male, 2 = female| protected attibute (see datasheet)|

### 7.2 Datasheet

A paragraph each, following Gebru et al. (2021).

**Why does it exist, and who made it? Who is represented, and who is excluded?**

TODO

**How was it collected and with what consent, and what was done to it before
you got it?**

TODO

**What may it be used for, and name one use that would be inappropriate.**

TODO

**Known biases and limitations. Men default at 24.2% and women at 20.8% before
any model exists — include that, and say why documenting it is a governance
task rather than a modelling one.**

TODO

---

## Section 8 — Exercises

8.1 is Section 7 above, so it is already done. Attempt 8.2 and 8.3, then do
*either* 8.4 *or* 8.5 — not both.

**8.2 Do the `-2` and `0` codes behave the same as each other? Two sentences
either way, with the evidence you used.**

One thing that immediately jumped out at me is that the whole lower quartile is 0 for the
`BILL_AMT`
and `PAY_AMT1` when the `PAY_0` is -2. That does not hold when the `PAY_0` is is 0

The next check I made was to compare the those months with the previous months. This also jumped out
at me. In both cases, the value was extremely likly to stay the same between months.

I am not sure what this suggests, but it certainly seems to suggest that those values do not mean
the same thing

**8.3 Your feature from the six bill columns, its correlation with the target,
and whether it beat `PAY_0`'s 0.325. If it did not, say whether you would keep
it anyway.**

TODO

**8.4 or 8.5 — say which one you did, then answer it.**

TODO

---

## Before you build the zip

**Use of AI assistants.** Say what you used one for, if anything, and what you
checked yourself. Using one is allowed; not being able to explain your own
submission is the problem. "None" is a complete answer.

TODO

**Anything you want to flag.** Optional — if something went wrong, if you ran
out of time, or if you want a second opinion on a judgement call. Delete the
TODO either way.

TODO
