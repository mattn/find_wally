# find_wally

Find Wally from given image file using Tensorflow.

## Create Model

```
$ git clone https://github.com/tensorflow/models
$ export PYTHONPATH=$PYTHONPATH:/path/to/models/research/object_detection;/path/to/models/research/slim;/path/to/models/research
```

Create labelmap.pbtxt

```
item {
  name: 'wally'
  id: 1
}
```

```
$ python /path/to/models/research/object_detection/dataset_tools/create_pet_tf_record.py --label_map_path=trained_model/labels.pbtxt --data_dir=. --output_dir=trained_model  
$ python /path/to/models/research/object_detection/legacy/train.py -train_dir=trained_model --pipeline_config_path=wally.config
```

See also https://towardsdatascience.com/how-to-find-wally-neural-network-eddbb20b0b90

## Usage

```
$ find_wally -model /path/to/tensorflow-object-detection-model.pb wally-evaluate.jpg
```

## Requirements

Tensorflow 1.9

## Installation

```
$ go get github.com/mattn/find_wally
```

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)
