# Dialectal Arabic POS Tagger 
Dialectal Arabic POS Tagger is a freeware module developed by the ALT team at Qatar Computing Research Institute (QCRI) to process Dialectal Arabic. The tagger was trained on a collection of dialectal Arabic tweets collected from frour regions - Egypt, Gulf, Maghrib and Levantine.
 
Arabic Dialects POS Tagger implemented using Keras/BiLSTM/ChainCRF. 

# Requirements

This segmenter requires the following packages:

- Python 3 (python2.7 should work with some minor changes)
    
- `tensorflow` version 0.9 or later: https://www.tensorflow.org
- `keras` version 1.2.2 or later: http://keras.io

## Installation

You can install the Dialectal Arabic POS Tagger by cloning the repo:

### Installing Dialectal Arabic POS Tagger from github
Clone the repo from the github using the following command:
```
git clone https://github.com/qcri/dialectal_arabic_pos_tagger
```
Or download the compressed file of the project, extract it.

## Getting started
Dialectal Arabic POS Tagger reads an input Arabic text file and produces the POS tags, one segment per line. The tagger expects the input file encoded in ``UTF-8``,
```
python arabic_pos_tagger.py -i [in-file] -o [out-file] 
```

For more details see:

``` 
python arabic_pos_tagger.py -h
```


## Publications

Randah Alharbi, Walid Magdy, Kareem Darwish, Ahmed Abdelali and Hamdy Mubarak. (2018) [Part-of-Speech Tagging for Arabic Gulf Dialect Using Bi-LSTM](http://www.lrec-conf.org/proceedings/lrec2018/pdf/483.pdf). Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). May 7-12, 2018. Miyazaki, Japan. Pages 3925-3932.

Kareem Darwish, Hamdy Mubarak, Ahmed Abdelali, Mohamed Eldesouki, Younes Samih, Randah Alharbi, Mohammed Attia, Walid Magdy and Laura Kallmeyer. (2018) [Multi-Dialect Arabic POS Tagging: A CRF Approach](http://www.lrec-conf.org/proceedings/lrec2018/pdf/562.pdf). Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). May 7-12, 2018. Miyazaki, Japan. Pages 93-98.



## Support

You can ask questions and join the development discussion:

- On the [Dialectal Arabic Tools Google group](https://groups.google.com/forum/#!forum/dat-users).
- On the [Dialectal Arabic Tools Slack channel](https://datsteam.slack.com). Use [this link](https://dat-slack-autojoin.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [Github issues](https://github.com/qcri/dialectal_arabic_tools/issues). Make sure to read [our guidelines](https://github.com/qcri/dialectal_arabic_pos_tagger/blob/master/CONTRIBUTING.md) first.


## License

Dialectal Arabic POS Tagger is covered by the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).


------------------