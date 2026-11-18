# Recorded measurements used by the quality tests

The two small compressed files contain nine measured columns and every source row. Tests read these local files; they never download data.

| Dataset | Rows | Columns retained | Attribution |
| --- | ---: | --- | --- |
| [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) | 17,379 | rental count, temperature, humidity, wind speed | Hadi Fanaee-T (2013), [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894) |
| [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) | 19,735 | appliance/light energy use, outdoor temperature, humidity, wind speed | Luis Candanedo (2017), [doi:10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G) |

Both datasets are licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The sources retain their dataset licenses; they are not relicensed under this repository's software license.

Modifications: select the named columns, retain their original numeric text and all rows in source order, omit the other columns, compress as CSV. The energy dataset's `rv1`/`rv2` columns are artificial random variables and are excluded. No rows were filled, dropped, shuffled or generated.

[manifest.json](manifest.json) pins each source download, source member, archive/source/derived SHA256, row count, columns and credit. The actual Bike Sharing CSV contains 17,379 rows.

To reproduce the files in a **new** directory, use the project's Python 3.12 environment.

```sh
.venv/bin/python scripts/fetch_quality_data.py --output /tmp/reproduced-quality-data
```

The command requires network access and rejects changed source or derived bytes. Python/zlib changes can alter compressed bytes; inspect differences before updating hashes. Normal tests need no network and reject corrupted fixtures before use.
