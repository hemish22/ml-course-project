# Benchmark labels

`queries.json` holds 86 hand-written search queries over 9 public-domain NASA
"This Week @NASA" episodes. Each row is `{video, query, start, end}`: the query
and the time interval (seconds) of its true answer.

Intervals were derived from the Whisper transcripts of the videos (segment
boundaries), then the query was written as a short description of that part of the
episode. The labels therefore lean toward what is *said*, which favours the speech
signal; this bias is documented in the report.

Fetch and index the videos with `scripts/fetch_demo.py` (the long episode) and
`scripts/fetch_benchmark.py`, then `python scripts/ingest.py --dir data/raw`.
