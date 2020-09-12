Silent Cities AudioTagging
--

Applying a pretrained DL model to annotate soundscapes. 
This was developed for analyzing the data collected in the [Silent Cities project](https://osf.io/h285u/).

Used model: 
- Audio Tagging on [Audioset](https://research.google.com/audioset/) using ResNet22 from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn)

Requirements
--
- [pytorch](https://pytorch.org/) 1.4.0
- [librosa](https://librosa.github.io/librosa/)
- tqdm
- pandas
- numpy
- matplotlib
- plotly

Usage
--
    python tag_silentcities.py [-h] [--length LENGTH] 
                       [--folder FOLDER] [--file FILE] [--verbose]
                       [--overwrite] [--out OUT]

    Silent City Audio Tagging with pretrained LeeNet11 on Audioset

    optional arguments:
    -h, --help       show this help message and exit
    --length LENGTH  Segment length
    --folder FOLDER  Path to folder with wavefiles, will walk through subfolders
    --file FILE      Path to file to process
    --verbose        Verbose (default False = nothing printed)
    --overwrite      Overwrite files (default False)
    --out OUT        Output file (pandas pickle), default is output.xz

This will save a pandas dataframe as an output file.

A heatmap can be generated using the function in [analysis.py](analysis.py) and [postprocessing.py](postprocess.py), to generate a heatmap such as this one : 

![Audio tagging of one night long recording in a street of Toulouse, France (March 16th / 17th 2020). Audio tagging was performed using a deep neural network pretrained on the Audioset dataset.
](silentcity.png)

Use the [make_interactive_pdf](postprocess.py) function to generate an estimate of probability densities at various scales, such as this one : 

![Audio tagging of one night long recording in a street of Toulouse, France (March 16th / 17th 2020). Audio tagging was performed using a deep neural network pretrained on the Audioset dataset.
](silentcity2.png)


Credits
--
Nicolas Farrugia, Nicolas Pajusco, IMT Atlantique, 2020. 

Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)