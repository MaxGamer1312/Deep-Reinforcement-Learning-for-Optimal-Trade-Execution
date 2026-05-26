import pandas as pd
import zstandard as zstd

dctx = zstd.ZstdDecompressor()
with open('../data/XNAS-20260524-6NCFWMPDHH/xnas-itch-20240523-20260522.mbo.csv.zst', 'rb') as compressed_file:
    with dctx.stream_reader(compressed_file) as reader:
        df = pd.read_csv(reader, nrows=100000)
print(df.columns)
df['ts_event'] = pd.to_datetime(df['ts_event'])
valid_dates = df[df['ts_event'] + pd.Timedelta(days=5) <= df['ts_event'].max()]['ts_event'].unique()
print(df['action'].unique())
print(df['side'].unique())