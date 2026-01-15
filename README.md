# Understanding or Memorizing? A Case Study of German Definite Articles in Language Models
> Jonathan Drechsel, Erisa Bytyqi, Steffen Herbold

[![arXiV](https://img.shields.io/badge/arXiv-2601.09313-blue.svg)](https://arxiv.org/abs/2601.09313)
[![arXiv](https://img.shields.io/badge/arXiv-2502.01406-blue.svg)](https://arxiv.org/abs/2502.01406)


This is a fork of the official source code for the training and evaluation of [GRADIEND](https://github.com/aieng-lab/gradiend), with adaptions required for our case study of German definite articles in language models.

## Quick Links

- [Understanding or Memorizing Paper](https://arxiv.org/abs/2601.09313)
- [GRADIEND Paper](https://arxiv.org/abs/2502.01406)
- German GRADIEND Training and Evaluation Datasets (Hugging Face):
  - [aieng-lab/de-gender-case-articles](https://huggingface.co/datasets/aieng-lab/de-gender-case-articles)
  - [aieng-lab/wortschatz-leipzig-de-grammar-neutral](https://huggingface.co/datasets/aieng-lab/wortschatz-leipzig-de-grammar-neutral)
- [SuperGLEBer Repository Fork](https://github.com/aieng-lab/SuperGLEBer) including our adaptions like bootstrap evaluation 

## Install

### Environment

```bash
conda create -f environment.yml python=3.9 
conda activate gradiend-de
```

The `environment.yml` file includes dependencies including build hash versions to ensure full reproducibility. In case of issues, please use `environment_no_builds.yml` instead, which omits the build hashes.
We further provide a `requirements.txt` file for pip installations.

### Datasets

While the datasets necessary to train and evaluate our models are provided via Hugging Face, we also provide scripts to recreate our datasets. 
To generate these, you also need to install spaCy's German model and download the Leipzig corpus:
```bash
python -m spacy download de_core_news_sm

mkdir -p data/der_die_das && cd data/der_die_das
wget https://downloads.wortschatz-leipzig.de/corpora/deu_news_2024_300K.tar.gz
tar -xvzf deu_news_2024_300K.tar.gz
cd ../..
```



## Usage
To train and evaluate a GRADIEND model for grammatical gender in German, run (for instance):

```bash
python train.py mode=gradiend_encoder pairing=N_MF num_runs=2
```
Here, the pairing N_MF refers to Nominative-Female-Male transition (corresponding to a der<->die transition).

**Most important Arguments:**
- `--mode`: Specifies the training method to use (default is `gradiend_encoder`, see `conf/mode/` for options).
- `--pairing`: Selects the grammatical gender pairs (options see below, e.g., MF, FN, MN, MFN, N_MF, NG_F, ....).
- `--mode.model_config.num_runs`: Sets how many models to train.

This ensures you have the required version for grammatical gender support.
You can override any configuration in the `/conf` directory. The example command above demonstrates a minimalist setup to train two GRADIEND models for the nominative male-female gender-case transition, using the original GRADIEND method.

### Gender Case Pairings

We identify a specific gender case pairing by a string code, starting with a case code (N=Nominative, A=Accusative, D=Dative, G=Genitive), followed by gender code (M=Male, F=Female, N=Neutral).
Depending on the kind of transition, we have the following options:

- **Gender Transitions** of genders X, Y at fixed case C: C_XY (e.g., A_FN, N_FN, G_MF)
- **Case Transitions** of cases X, Y at fixed gender G: XY_G (e.g., ND_M, GA_F, NG_N)
- **Whole Gender Transition** of genders X, Y across all cases: XY (e.g., MF) 
- **Whole Case Transition** of cases X, Y across all genders: XY (e.g., ND) 

The latter two cases are not used in the paper, but are supported for further experiments.

Within gender and case code, the genders/cases are ordered lexicographically for gender and by their case number for cases (N=1, G=2, D=3, A=4).


## Reproducing Results from the Paper

We use `RESULTS_DIR="results"` and `IMG_DIR="img"` as base directories for results and images, respectively (you can change these in `gradiend.util.py`).
All scripts are to be run from the base folder of this repository.

### Training

- Train custom MLM heads for decoder-only models.
  - Run `gradiend.training.decoder_only_mlm.training_de.py`, which trains the MLM heads for a range of pooling_lengths for `german-gpt2` and `Llama-3.2-3B` (output in `{RESULTS_DIR}/decoder-mlm-head-gender-de/AF-AM-AN-DF-DM-DN-GF-GM-GN-NF-NM-NN/`)
  - This script also creates a visualization of the MLM head performance for different pooling lengths in `{IMG_DIR}/de_pooling_length_comparison.pdf` 
- Train all GRADIEND models
  - Run `shell_scripts/train_all_interesting_gradiends_{model_id}.sh` to train all GRADIEND variants of that model as used in the paper.
    - Available model_ids are 
      - bert ([`bert-base-german-cased`](https://huggingface.co/google-bert/bert-base-german-cased))
      - euro ([EuroBERT-210m](https://huggingface.co/EuroBERT/EuroBERT-210m))
      - gbert ([deepset/gbert-large](https://huggingface.co/deepset/gbert-large))
      - gpt ([dbmdz/german.gpt2](https://huggingface.co/dbmdz/german-gpt2))
      - modern_bert ([LSX-UniWue/ModernGBERT_1B](https://huggingface.co/LSX-UniWue/ModernGBERT_1B))
      - llama ([meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))
    - Outputs in `{RESULTS_DIR}/experiments/gradiend/{pairing}_neutral_augmented/dim_1_inv_gradient/{model_id}/{i}` for `i` in {0,1,2} (three runs with different seeds)
      - Notice that after training of all seeds, only the best run is kept to save space.
      - You can use `GradiendModel.from_pretrained(...)` from the base folder (without `i`), which automatically selects the best run.
      - The base folder also contains various files with encoded values and their statistics for different splits.
    - Notice that the Llama model requires 3 A100 GPUs due to its size.
  - Optional: Run `shell_scripts/train_all_gradiends.sh` to train even more GRADIEND variants (not reported in the paper, e.g., `MN` to create a male-neutral GRADIEND across all cases).

### Evaluation

- The training already concludes with computation of the encoder (i.e., the encoder analysis does not need to be run separately).
  - Use `gradiend.evauation.encoder.de_encoder_plot.py` to create encoding plots (output in `{IMG_DIR}/encoded_values_stacked_models/`).
  - Use `gradiend.export.de_encoder_table.py` to print a table of Pearson correlations of all trained models.
- Use `gradiend.evaluation.decoder.de_decoder_probabilities.py` to compute probability analyses.
  - Then, you can use `gradiend.evaluation.decoder.de_decoder_probability_analysis.py` to assess statistics about probability change
    - This script prints the statistisc as copybable LaTeX table
    - It also creates the GRADIEND-modified models saved in `{RESULTS_DIR}/changed_models` for SuperGLEBer evaluation
  - Use `gradiend.evaluation.de_decoder_probability_heatmap.py` to create heatmaps of probability changes (output in `{IMG_DIR}/decoder_heatmaps_multi/`)
- Run `gradiend.evaluation.xai.venn_plot.py` to create Venn diagrams of top-k token overlaps (output in `{IMG_DIR}/venn/`).
  - Run `*.venn_plot_multi.py` to plot these as subplots into wide compact figures (output in `{IMG_DIR}/venn/across_models_...`).
  - Run `gradiend.evaluation.xai.venn_ablation_multi.py` to create all ablation plots varying k at once (output in `{IMG_DIR}/intersection_subsets/`).
  - Run 'gradiend.evaluation.xai.max_overlap.py' to compute and print the maximum top k overlaps across all models and article groups as LaTeX table.
- Use our [SuperGLEBer fork](https://github.com/aieng-lab/SuperGLEBer) to evaluate our GRADIEND-modified models on the SuperGLUE benchmark, including bootstrap evaluation.


### Data Generation

- Run `gradiend.data.german_articles.py` to generate the German GRADIEND [gender-case article](https://huggingface.co/datasets/aieng-lab/de-gender-case-articles) dataset.
- Run `gradiend.data.german_neutral.py` to generate the [grammar neutral dataset](https://huggingface.co/datasets/aieng-lab/wortschatz-leipzig-de-grammar-neutral) from the Leipzig corpus.


## Main Changes to the Base Repository

To keep track of the changes made to the original GRADIEND repository, here is a summary of the main additions and modifications:

- German Training Data (generation and access)
- Fine-grained GRADIEND training for two given gender-case cells
  - `gradiend.evaluation.xai.io.GradiendGenderCaseConfiguration` is a utility class that defines easy definition of a specific configuration, including various functions, e.g., to derive datasets, articles, ...
- Probability analysis (`gradiend.evaluation.decoder.de_decoder_probability_analysis`, ...)
- Top-k analysis (`gradiend.evaluation.xai.venn_plot`, ...)


## Citation

Please cite the following paper if you use our code or datasets in your research:

```bib
@misc{drechsel2026understandingmemorizingcasestudy,
      title={Understanding or Memorizing? A Case Study of German Definite Articles in Language Models}, 
      author={Jonathan Drechsel and Erisa Bytyqi and Steffen Herbold},
      year={2026},
      eprint={2601.09313},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09313}, 
}
```

The GRADIEND method itself is described in:

```bib
@misc{drechsel2025gradiendfeaturelearning,
      title={{GRADIEND}: Feature Learning within Neural Networks Exemplified through Biases}, 
      author={Jonathan Drechsel and Steffen Herbold},
      year={2025},
      eprint={2502.01406},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.01406}, 
}
```