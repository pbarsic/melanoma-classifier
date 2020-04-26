Melanoma Classifier 
> A PyTorch C++ classifier to detect cancerous lesions.

```
 pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
PYTHONPATH=$(pwd) src/train.py 
```

```
PYTHONPATH=$(pwd) src/export_to_onnx.py --input_model_path /home/alex/runs/melanoma-model/model-0.673.pt --save_dir /home/alex/runs/melanoma-model/model-0.673.onnx
```

The way that Alex intends to run this in the browser is to follow [this link](https://microsoft.github.io/onnxjs-demo/#/resnet50).
