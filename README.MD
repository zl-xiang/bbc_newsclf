# CMT316 Coursework I Part 2

- [**Project Repository**](https://bitbucket.org/4r2eBurger/news_clf/src/master/)
- [**Project Report**](https://bitbucket.org/4r2eBurger/news_clf/src/master/report_latex/CWI_part2_report.pdf)

## File Description

### Source Code

- [`text_processing.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/text_processing.py): A module which consists of plain text processing functions, such as POS counting, TF-IDF weights calculation and etc.

- [`prepocessing.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/prepocessing.py): A module which is used to process raw data and tranform them into sample vectors that can be directly used for training.

- [`one_shot_eval.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/one_shot_eval.py): A module that conducts simple train-dev-test evaluation.

- [`fs_n_training.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/fs_n_training.py): A module that conducts feature selection, grid-search with cross-validation.

- [`main.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/main.py): A moudule that combines steps of processing-training-validation-evaluation pipeline.

- [`main.ipynb`](https://bitbucket.org/4r2eBurger/news_clf/src/master/main.ipynb): processing-training-validation-evaluation pipeline demo.

- [`unseen_test.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/unseen_test.py): A module that consists simple evaluation of trained models by predicting on up-to-date unseen BBC news documents.

- [`utils.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/utils.py): A module that includes simple file IO functions.

- [`sLDA_test.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/sLDA_test.py): (optional) Test case of supervised topic modelling, implemented with [Tomotopy Library](https://github.com/bab2min/tomotopy).

- [`sLDA_test.ipynb`](https://bitbucket.org/4r2eBurger/news_clf/src/master/sLDA_test.ipynb): (optional) supervised topic modelling demo.

- [`LDA_test.py`](https://bitbucket.org/4r2eBurger/news_clf/src/master/LDA_test.py): (optional) LDA topic modelling test case.

### Dataset

- `dataset`(DIR): Raw dataset that is used in experimental pipeline.

- `unseen_dat`(DIR): Raw dataset that is used in trained models evaluation.

### Cached Files

- `SVC_model.pickle`: Serialisation of best fine-tuned SVC model.

- `softmax_model.pickle`: Serialisation of best fine-tuned Softmax Regression model.

- `vocab.pickle`: Serialisation of news contents vocabulary.

- `vocab_title.pickle`: Serialisation of news headlines vocabulary.

- `X_1000_300.pickle`: Serialisation of sample set X.

- `y_1000_300.pickle`: Serialisation of sample set y.

## Get Started

### Prerequisites

**OS**: OSX/Linux

Needs to be run in [**Conda**](https://docs.conda.io/projects/conda/en/latest/index.html) environment.

#### Conda dependencies

- python 3 `conda install python=3.9`

- scikit-learn 0.23.2  `conda install -c conda-forge scikit-learn=0.23.2` 

- numpy `conda install numpy`

- pandas `conda install pandas`

- nltk `conda install nltk`

- matplotlib `conda install matplotlib`

#### Pip dependencies

- [Tomotopy](https://github.com/bab2min/tomotopy): `pip install tomotopy` (in conda environment)

### Steps

1. Clone or download project

2. Enter the source directory by `cd [user DIR]/news_clf`

3. create new conda running environment : `conda create --name news_clf python=3.9 pandas numpy scikit-learn=0.23.2 nltk matplotlib pip`

4. activate running environment `conda activate news_clf`

#### One-shot Train-Dev-Test evaluation

Run `python one_shot_eval.py`

#### Feature selection, Training and GridSearch with Cross-validation evaluation

Run `python main.py` or run jupyter notebook demo `main.ipynb`

#### Test on up-to-date small samples

Run `python unseen_test.py`

#### Demo of sLDA topic modelling with linear SVC (Optional)

Run `python sLDA_test.py` or run jupyter notebook demo `sLDA_test.ipynb`

* Tested on OSX and Ubuntu container, might fail in Linux due to [unsolved bug](https://github.com/bab2min/tomotopy#history) of tomotopy library.
