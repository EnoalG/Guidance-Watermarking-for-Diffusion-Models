from os.path import dirname, join as joinpath

MODELDIR = joinpath(dirname(__file__), '../detector/models')
WMDIR = joinpath(dirname(__file__), '../watermarking/models')

IMTESTDIR = joinpath(dirname(__file__), './test_images')
DATATESTDIR = joinpath(dirname(__file__), './test_data')
LOCALDATA = joinpath(dirname(__file__), './__local_data__')

WMDATASET =  '/path/to/guided-diffusion/images/sd2_model/'
FLICKRSET = '/path/to/Flickr/images/0'
MSCOCO =  '/path/to/datasets/mscoco/annotations/pure_captions_train2014.jsonl'

BENCHMARK_DIR= joinpath(dirname(__file__), './__local_data__/results_benchmark/')