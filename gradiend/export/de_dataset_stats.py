from gradiend.data import read_article_ds, read_de_neutral


de_neutral = read_de_neutral()
print(len(de_neutral), "examples in de_neutral dataset")
exit(1)

articles = [
    'NM', 'NF', 'NN',
    'AM', 'AF', 'AN',
    'DM', 'DF', 'DN',
    'GM', 'GF', 'GN',
]

def latex_dataset_name(article):
    # maps NM -> \dataNM etc.
    return rf"\data{article}"

rows = []

for article in articles:
    splits = {
        split: read_article_ds(article, split=split)
        for split in ["train", "validation", "test"]
    }

    n_train = len(splits["train"])
    n_val   = len(splits["validation"])
    n_test  = len(splits["test"])
    n_total = n_train + n_val + n_test

    rows.append(
        f"{latex_dataset_name(article)} & {article} & "
        f"{n_total} & {n_train} & {n_val} & {n_test} \\\\"
    )

# print LaTeX table body
print("\n".join(rows))
