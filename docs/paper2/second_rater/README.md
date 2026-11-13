# Independent raters (Paper 2: Methods, and the agreement table in Results)

Two research assistants rated the same 41 configurations alone, blind to the first author's
answers and to each other. Their identities are not tied to their ratings here.

| File | Content |
|---|---|
| `sheet.md` | The sheet the raters received: the nine questions and the 41 configurations |
| `key.csv` | The first author's rating of each item (F flagged, P partly, N not flagged) and the questions named |
| `ratings_A.csv`, `ratings_B.csv` | Each rater's returned answers, transcribed unchanged |

Results:

```
python 9-Paper2/second_rater_agreement.py A
python 9-Paper2/second_rater_agreement.py B
python 9-Paper2/second_rater_agreement.py --between A B
python 9-Paper2/second_rater_decomposition.py A B
```

These write `5-Results/paper2__second-rater-agreement__{A,B,A-vs-B}.csv`,
`paper2__second-rater-by-mechanism.csv` and `paper2__second-rater-decomposition.csv`.
`9-Paper2/second_rater_consistency.py docs/paper2/second_rater` prints the post-hoc checks.
