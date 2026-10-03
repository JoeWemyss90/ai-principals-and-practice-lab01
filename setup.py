# setup.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("UCI_Credit_Card.csv")
print(df.shape)  # (30000, 25)
print(df["default.payment.next.month"].mean().round(3))  # ~0.221 default rate

# THIS IS THE CODEBOOK FOR THE DATASET
# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:␍
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.␍
# X2: Gender (1 = male; 2 = female).␍
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).␍
# X4: Marital status (1 = married; 2 = single; 3 = others).␍
# X5: Age (year).␍
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005.
# The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.␍
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. ␍
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.␍

# section1_profile.py
print(df.dtypes.value_counts())
# print(df.isna().sum())  # new DF showing which fields have blank
print(df.isna().sum().sum())  # total blanks in the whole file
print([c for c in df.columns if c.startswith("PAY_") and "AMT" not in c])

# section1_audit.py
for c in ["SEX", "EDUCATION", "MARRIAGE"]:
    print(f"{c:10s} {sorted(df[c].unique().tolist())}")
print(f"{'PAY_0':10s} {sorted(df['PAY_0'].unique().tolist())}")


print("EDUCATION in {0,5,6}:", df.EDUCATION.isin([0, 5, 6]).sum())
print("MARRIAGE == 0      :", (df.MARRIAGE == 0).sum())
print(
    "PAY_0 in {-2, 0}   :",
    df.PAY_0.isin([-2, 0]).sum(),
    f"({100 * df.PAY_0.isin([-2, 0]).mean():.1f}%)",
)

rate = df.groupby("PAY_0")["default.payment.next.month"].agg(["mean", "size"])
print(rate.assign(mean=rate["mean"].round(3)).to_string())


# section2_centre.py
for c in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]:
    s = df[c]
    print(
        f"{c:10s} mean={s.mean():>10,.0f}  median={s.median():>9,.0f}  "
        f"ratio={s.mean() / s.median():.2f}"
    )

pd.set_option("display.max_columns", None)

# print(df.nlargest(20, "PAY_AMT1"))
# print(df.nlargest(20, "BILL_AMT1"))

# section2_spread.py
for c in ["LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"]:
    s = df[c]
    q1, q3 = s.quantile([0.25, 0.75])
    print(f"{c:10s} sd={s.std():>10,.0f}   IQR={q3 - q1:>10,.0f}")


# section2_shape.py
for c in ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]:
    print(
        f"{c:10s} skew={stats.skew(df[c]):>6.2f}   "
        f"excess kurtosis={stats.kurtosis(df[c]):>8.2f}"
    )


# section2_plots.py
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df.AGE, bins=40)
axes[0].set_title("AGE: roughly bell-shaped")
axes[1].hist(df.PAY_AMT1 + 1, bins=100)
axes[1].set_title("PAY_AMT1: one bar and a long tail")
axes[1].set_xscale("log")
axes[2].boxplot(df.LIMIT_BAL, vert=True)
axes[2].set_title("LIMIT_BAL: box plot")
plt.tight_layout()
plt.xscale("log")
# plt.show()


# section3_correlation.py
for a, b in [
    ("BILL_AMT1", "PAY_AMT1"),
    ("LIMIT_BAL", "BILL_AMT1"),
    ("LIMIT_BAL", "AGE"),
]:
    p = stats.pearsonr(df[a], df[b]).statistic
    s = stats.spearmanr(df[a], df[b]).statistic
    print(f"{a:10s} vs {b:10s}  Pearson={p:6.3f}   Spearman={s:6.3f}")


bills = [f"BILL_AMT{i}" for i in range(1, 7)]
print(df[bills].corr().round(2).to_string())

c = (
    df.drop(columns=["ID"])
    .corr(numeric_only=True)["default.payment.next.month"]
    .drop("default.payment.next.month")
    .sort_values(key=abs, ascending=False)
)
print(c.head(8).round(3).to_string())

# section4_scaling.py

X = df[["AGE", "LIMIT_BAL", "BILL_AMT1"]]
print(X.agg(["min", "max"]).to_string())
print(StandardScaler().fit_transform(X)[:3].round(2))

# section4_missing.py
d = df.rename(
    columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"}
).copy()
d["EDUCATION"] = d["EDUCATION"].replace({0: np.nan, 5: np.nan, 6: np.nan})
d["MARRIAGE"] = d["MARRIAGE"].replace({0: np.nan})
print(d[["EDUCATION", "MARRIAGE"]].isna().sum())
print("rows affected:", d[["EDUCATION", "MARRIAGE"]].isna().any(axis=1).sum())

print(
    f"EDUCATION missing -> default rate {d.loc[d.EDUCATION.isna(), 'default'].mean():.3f}"
)
print(
    f"EDUCATION present -> default rate {d.loc[d.EDUCATION.notna(), 'default'].mean():.3f}"
)
print(
    f"MARRIAGE  missing -> default rate {d.loc[d.MARRIAGE.isna(), 'default'].mean():.3f}"
)
print(
    f"MARRIAGE  present -> default rate {d.loc[d.MARRIAGE.notna(), 'default'].mean():.3f}"
)


d["EDUCATION_was_missing"] = d["EDUCATION"].isna().astype(int)
d["MARRIAGE_was_missing"] = d["MARRIAGE"].isna().astype(int)


# section4_outliers.py
s = df.BILL_AMT1
q1, q3 = s.quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
z = (s - s.mean()) / s.std()

print(f"IQR fences : {lo:>12,.0f} to {hi:>12,.0f}")
print(
    f"IQR flags  : {((s < lo) | (s > hi)).sum():>6,}  ({100 * ((s < lo) | (s > hi)).mean():.1f}%)"
)
print(f"|z| > 3    : {(z.abs() > 3).sum():>6,}  ({100 * (z.abs() > 3).mean():.2f}%)")


print("negative bills:", (s < 0).sum(), " min:", f"{s.min():,.0f}")

print("exact duplicate rows (ignoring ID):", df.drop(columns=["ID"]).duplicated().sum())


# section5_smote.py

d = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
num = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)
X, y = d[num], d["default"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)  # SPLIT FIRST
print("train before:", dict(sorted(Counter(y_tr).items())))
X_aug, y_aug = SMOTE(random_state=0).fit_resample(X_tr, y_tr)  # augment TRAIN only
print("train after :", dict(sorted(Counter(y_aug).items())))
print("train rows  :", len(y_tr), "->", len(y_aug))


plain = Pipeline(
    [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]
).fit(X_tr, y_tr)
smote = ImbPipeline(
    [
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=0)),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
).fit(X_tr, y_tr)

for name, m in [("no SMOTE", plain), ("SMOTE", smote)]:
    p = m.predict(X_te)
    print(
        f"{name:9s} acc={accuracy_score(y_te, p):.3f}  "
        f"recall={recall_score(y_te, p):.3f}  precision={precision_score(y_te, p):.3f}"
    )

# section6_leakage.py

X_all, y_all = SMOTE(random_state=0).fit_resample(X, y)  # resample EVERYTHING first
wrong = cross_val_score(
    Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000))]),
    X_all,
    y_all,
    cv=5,
    scoring="f1",
)
print(f"WRONG  F1 per fold {wrong.round(3)}  mean={wrong.mean():.3f}")

right = cross_val_score(
    ImbPipeline(
        [
            ("s", StandardScaler()),
            ("sm", SMOTE(random_state=0)),
            ("c", LogisticRegression(max_iter=2000)),
        ]
    ),
    X,
    y,
    cv=5,
    scoring="f1",
)
print(f"RIGHT  F1 per fold {right.round(3)}  mean={right.mean():.3f}")

# section6_pipeline.py

d = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"}).drop(
    columns=["ID"]
)
d["EDUCATION"] = d["EDUCATION"].replace({0: np.nan, 5: np.nan, 6: np.nan})
d["MARRIAGE"] = d["MARRIAGE"].replace({0: np.nan})

categorical = ["SEX", "EDUCATION", "MARRIAGE"]
numeric = (
    ["LIMIT_BAL", "AGE"]
    + [f"PAY_{i}" for i in range(1, 7)]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

prep = ColumnTransformer(
    [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "impute",
                        SimpleImputer(strategy="most_frequent", add_indicator=True),
                    ),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical,
        ),
    ]
)
model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])

X, y = d[numeric + categorical], d["default"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)
model.fit(X_tr, y_tr)  # every step learns from training rows only
print(
    "features after preprocessing:", model.named_steps["prep"].transform(X_tr).shape[1]
)
print("test accuracy:", round(model.score(X_te, y_te), 3))

# exercise_8_2.py
# your implementation:
# - for clients with PAY_0 == -2, what do BILL_AMT1 and PAY_AMT1 look like?
# - and for PAY_0 == 0? do the two codes behave the same as each other?
# - does your -2 hypothesis survive? write two sentences either way

pay_minus_2 = df.loc[df["PAY_0"] == -2, ["BILL_AMT1", "PAY_AMT1"]]

print(pay_minus_2.describe())


pay_0 = df.loc[df["PAY_0"] == 0, ["BILL_AMT1", "PAY_AMT1"]]

print(pay_0.describe())

print("PAY_0 == -2")
print(df.loc[df["PAY_0"] == -2, "PAY_2"].value_counts(normalize=True).sort_index())

print("\nPAY_0 == 0")
print(df.loc[df["PAY_0"] == 0, "PAY_2"].value_counts(normalize=True).sort_index())
