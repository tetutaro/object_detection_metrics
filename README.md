# object_detection_metrics

calculate metrics used to evaluate object detection algorithms.

## metrics

- micro mean Average Precision (micro-mAP):
    - calculate True-Positives and False-Positives for each object category
    - collect True-Positives and False-Positives in all categories
    - calculate Average Precision (micro mean)
- macro mean Average Precision (macro-mAP):
    - calculate Average Precision for each object category
    - calculate mean of Average Precision in all categories (macro mean)
- weighted mean Average Precision (weighted-mAP)
    - calculate Average Precision for each object category
    - calculate weighted mean of Average Precision in all categories weighted by the number of ground truth in the category (weighted mean)

## features of this implementation

- the Ground Truth file and Prediction file to be read are in JSON Lines format.
    - create a Python tool that converts COCO-formatted JSON to the corresponding JSON Lines format.
- when reading the file, check the format with pydantic.
- perform Ground Truth vs Prediction IoU calculations at once using a numpy 3D array.
- uses an original algorithm to calculate Average Precision.

## install

`> pip install "git+https://github.com/tetutaro/object_detection_metrics.git"`

## usage

### convert Ground Truth/Prediction file in COCO format (JSON)

```
usage: convert_coco_annotations [-h] [--output OUTPUT] coco

convert coco annotations to jsonl format

positional arguments:
  coco             coco annotation file

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  output jsonl file
                   (default: os.path.basename(coco).replace('.json', '.jsonl')
```

### calculate metrics using the command line application

```
usage: object_detection_metrics [-h] --trues TRUES --preds PREDS [--output OUTPUT] [--verbose]

calculate metrics used to evaluate object detection

optional arguments:
  -h, --help            show this help message and exit
  --trues TRUES, -t TRUES
                        the file of ground truth bounding boxes
  --preds PREDS, -p PREDS
                        the file of predicted bounding boxes
  --output OUTPUT, -o OUTPUT
                        output filename (.jsonl) (if -o none, just print metrics)
  --verbose, -v         output AP for each category
```

### calculate metrics using the Python module

```
> from object_detection_metrics import Evaluator

> evaluator = Evaluator(
      trues='file path of Ground Truths',
      preds='file path of Predictions',
      verbose=True or False
  )
> evaluator.accumulate()
> print(evaluator.category_total.n_img)
Total # of images
> print(evaluator.category_total.n_true)
Total # of Ground Truths
> print(evaluator.category_total.n_pred)
Total # of Predictions
> print(evaluator.category_total.aps[75])
micro mean Average Precision at IoU threshold is 0.75
> print(evaluator.category_total.aps[100])
micro mean Average Precision at IoU threshold is 0.50:0.95:0.05
> print(evaluator.macro_maps[75])
macro mean Average Precision at IoU threshold is 0.75
> print(evaluator.macro_maps[100])
macro mean Average Precision at IoU threshold is 0.50:0.95:0.05
> print(evaluator.weighted_maps[75])
weighted mean Average Precision at IoU threshold is 0.75
> print(evaluator.weighted_maps[100])
weighted mean Average Precision at IoU threshold is 0.50:0.95:0.05
> print(evaluator.categories[<category ID>].n_img)
# of images of the cateogry ID
> print(evaluator.categories[<category ID>].n_true)
# of Ground Truths of the category ID
> print(evaluator.cateogries[<category ID>].n_pred)
# of Predictions of the category ID
> print(evaluator.categories[<category ID>].aps[75])
Average Precision of the category ID at IoU threshold is 0.75
> print(evaluator.categories[<category ID>].aps[100])
Average Precision of the category ID at IoU threshold is 0.50:0.95:0.05
```

## file format

- for both Ground Truth and Prediction, the file format must be JSON Lines
- the extension of file must be ".jsonl"
- one line (JSON) represents the information of one image
- images must not be duplicated in each of Ground Truth and Prediction file

### Ground Truth

(actually, the following JSON must be written in one line, but for the sake of clarity, line breaks and indent)

```
{
    "image_id": "the ID of image (ex. filename) to ident the image",
    \\ bouding boxes
    "bboxes": [
        {
            \\ category ID of detected object (int)
            "category_id": 1,
            \\ bouding box (min_x, min_y, max_x, max_y) in pixel
            "bbox": [10.1, 20.2, 30.3, 40.4]
        },
        ...
    ]
}
```

### Prediction

(actually, the following JSON must be written in one line, but for the sake of clarity, line breaks and indent)

```
{
    "image_id": "the ID of image (ex. filename) to ident the image",
    \\ bouding boxes
    "bboxes": [
        {
            \\ category ID of detected object (int)
            "category_id": 1,
            \\ bouding box (min_x, min_y, max_x, max_y) in pixel
            "bbox": [10.1, 20.2, 30.3, 40.4],
            \\ confidence score (float)
            "score": 0.85
        },
        ...
    ]
}
```
