### Training datasets and model provenance

The model-training datasets are documented below. They have different roles from
the videos a user uploads to the web application for searching.

#### Datasets used to train the pretrained encoders

These encoders were trained by their original authors. This project loads their
released weights and keeps them frozen; we did not retrain them on these corpora.

| Model used in this project | Original training dataset / corpus | Purpose in our pipeline |
|---|---|---|
| OpenCLIP ViT-B/32, `laion2b_s34b_b79k` | **LAION-2B English**, the approximately two-billion image–text-pair English subset of LAION-5B. | Provides aligned visual and text embeddings for matching queries to frames. |
| Whisper `base`, through faster-whisper | **OpenAI's 680,000-hour multilingual audio–transcript corpus**, collected from the internet. The faster-whisper checkpoint is a CTranslate2 conversion of the released Whisper base model. | Converts speech into timestamped text. |
| Sentence Transformer `all-MiniLM-L6-v2` | **A mixed sentence-pair training collection with over one billion pairs/tuples**. Sources include Reddit comments, S2ORC, WikiAnswers, PAQ, Stack Exchange, MS MARCO, and other datasets listed in the model card. This sentence-embedding training builds on a pretrained MiniLM backbone. | Encodes queries and transcript segments for semantic similarity. |

Sources: [OpenCLIP model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K),
[Whisper training-data description](https://github.com/openai/whisper/blob/main/model-card.md#training-data),
[faster-whisper base conversion](https://huggingface.co/Systran/faster-whisper-base),
[MiniLM sentence-embedding training data](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#training-data).

#### Dataset used to train or fit models in this project

Our own fitted models use the **labeled query–second relevance feature dataset**
stored in `results/ml/dataset.csv.gz`. Its annotations are in
`benchmark/queries.json`; `analysis/features.py` constructs the feature table.

| Models fitted by us | Training input | Training target / objective |
|---|---|---|
| Ridge, Random Forest, Gradient Boosting regressors | 31 engineered relevance features per query–second row. | Predict graded relevance (`rel_graded`). |
| Logistic Regression, k-nearest neighbours, Gradient Boosting classifiers | The same 31-feature dataset. | Predict whether the second belongs to the answer interval (`rel_binary`). |
| K-means and Gaussian mixture model | Eight selected features from the same dataset, standardized on the training fold. | Discover clusters without labels. Training labels subsequently calibrate cluster relevance. |
| Gaussian Hidden Markov Model | The same eight-feature subset, ordered into separate query–video sequences. | Learn Gaussian emissions and state transitions without labels. Training labels subsequently calibrate state relevance. |

**Dataset size and origin:** 28,900 labeled query–second rows, 86 annotated queries,
and nine source recordings from NASA news. There are 2,385 relevant rows (8.3%).
The source recordings are the material from which this training benchmark was
constructed. Their role here is established by the saved feature table and
annotations, rather than by which videos happen to appear in the web app.

**Training/evaluation split:** leave one entire source video out per fold. Each
model is fitted on rows from eight source videos and evaluated on the remaining
video, for nine folds. Preprocessing, model selection and relevance calibration
use the training fold only. The 28,900 rows describe the complete benchmark;
they are not all used for training in any one fold.

**Why this dataset:** the rows directly represent the relevance prediction task,
and their temporal order supports the HMM. The benchmark provides both graded
and binary labels, allowing the existing and new models to be compared. Labels
were derived from transcripts, so the benchmark favors speech and has limited
domain diversity.

**Web-app uploads:** ingestion extracts frames, transcripts and embeddings and
builds search indexes. Uploading a video does not fit these analysis models or
update the pretrained encoder weights. A source recording may also be available
as a demo, but that does not define the training dataset.

MSR-VTT and Charades appear in evaluation-harness code, but the saved experiments
have not been trained or evaluated on those datasets.
