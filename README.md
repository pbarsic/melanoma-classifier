# Melanoma Classifier 
> A PyTorch C++ classifier to detect cancerous lesions.

## Steps to set up dataset:

    1) download dataset from Kaggle and extract it into ~/datasets  
    2) Perform:  
```
      pushd ~/datasets/skin-cancer-mnist-ham10000/
      mkdir imgs
      popd
```

## Set up dev environment:

```
virtualenv m-class
source m-class/bin/activate
 pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
pip install -r requirements.txt
```

## Train the dataset:

```
PYTHONPATH=$(pwd) src/train.py 
```

## Export the model:  
```
PYTHONPATH=$(pwd) src/export_to_onnx.py --input_model_path /home/alex/runs/melanoma-model/model-0.673.pt --save_dir /home/alex/runs/melanoma-model/model-0.673.onnx
```

## Apply the module:  
The way that Alex intends to run this in the browser is to follow [this link](https://microsoft.github.io/onnxjs-demo/#/resnet50).


## Install node packages:
There are many ways to install Node.js. The best way is to use [NVM](https://github.com/nvm-sh/nvm), the Node Version Manager. This is similar to using Python virtual environments. It's what you want.
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
```
