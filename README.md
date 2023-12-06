Silent Cities Data preparation
--

This repository includes the python code that was used to prepare the [Silent Cities Dataset](https://osf.io/h285u/) prior to publication. It is provided here for transparency and documentation, but cannot be used as is. 

Description
--
The overall sequence of processing is as follows. 

Data preparation:
- Generation of a metadata file for each recording site, gathering all available audio for this site using [metadata_file.py](metadata_file.py)
- Generation of a metadata file across all sites using [metadata_site.py](metadata_site.py)

Data analysis (Script [audio_processing.py](audio_processing.py)):
- Calculation of ecoacoustic indices every 10 seconds
- Applying a pretrained Deep Learning model for audio tagging. (pretrained on [Audioset](https://research.google.com/audioset/) using ResNet22 from [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn))

Final step:
- Encoding of all raw audio files to FLAC format and final export in script [convert.py](convert.py)

Downloader
--
The script [download_osf.py](download_osf.py) will download the whole dataset. Beware that the full dataset is close to 5 terabytes. 

```
usage: download_osf.py [-h] [--dest DEST [DEST ...]]

options:
  -h, --help            show this help message and exit
  --dest DEST [DEST ...]
                        Add here as many paths as you want to download the data. If one path does not have enough space, it will switch to the next one
```

Example usage 
```
python download_osf.py --dest /media/disk2/ /media/disk1/
```
This will first download on `/media/disk1/`, then when it's full it will download on `/media/disk2/`. 



Requirements
--
A requirements file is provided for reference [here](requirements.txt), exact versions might not be needed. Please adapt according to your setup. 

Usage
--
## Preprocessing : 

file by file (individual files)

    python metadata_file.py [-h] [--folder FOLDER] 
                                [--save_path SAVE_PATH]
                                [--verbose] [--all]

    Silent City Meta data generator file by file

    optional arguments:
    -h, --help            show this help message and exit
    --folder FOLDER       Path to folder with wavefiles, will walk through subfolders
    --save_path SAVE_PATH
                            Path to save meta data
    --verbose             Verbose (default False = nothing printed)
    --all                 process all site. for multiple HDD (list path must be given in HDD variable in the script)
Generate one CSV by site

All sites

    python metadata_site.py [-h] [--folder FOLDER] 
                            [--save_path SAVE_PATH] [--database DATABASE] 
                            [--verbose]

    Silent City Meta data generator site by site

    optional arguments:
    -h, --help            show this help message and exit
    --folder FOLDER       Path to folder with meta data
    --save_path SAVE_PATH Path to save meta data by site
    --database DATABASE   database (csv)
    --verbose             Verbose (default False = nothing printed)
Generate metadata from all site (one CSV)

## Audio processing
**First download the ResNet22 pretrained model using instructions [here](https://github.com/qiuqiangkong/audioset_tagging_cnn#audio-tagging-using-pretrained-models)**

Processing one site

    python audio_processing.py [-h] [--length LENGTH] [--batch_size BATCH_SIZE] 
                                [--metadata_folder METADATA_FOLDER] [--site SITE] 
                                [--folder FOLDER] [--database DATABASE] [--nocuda]

    Silent City Audio Tagging with pretrained ResNet22 on Audioset (from https://github.com/qiuqiangkong/audioset_tagging_cnn)

    optional arguments:
    -h, --help            show this help message and exit
    --length LENGTH       Segment length
    --batch_size BATCH_SIZE
                            batch size
    --metadata_folder METADATA_FOLDER
                            folder with all metadata
    --site SITE           site to process
    --folder FOLDER       Path to folder with wavefiles, will walk through subfolders
    --database DATABASE   Path to metadata (given by metadata_site.py)
    --nocuda              Do not use the GPU for acceleration



<!-- 
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
 -->

Credits
--
Nicolas Farrugia, Nicolas Pajusco, IMT Atlantique, 2020. 

Code for Audioset Tagging CNN from [Qiu Qiang Kong](https://github.com/qiuqiangkong/audioset_tagging_cnn)

Code for Eco-acoustic indices from [Patric Guyot](https://github.com/patriceguyot/Acoustic_Indices)
