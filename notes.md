Chapter 1 : Introducing deep learning and the Pytorch Library
=============================================
1. Intro
    a) 'Artificial intelligence'
        #. Not 'thinking' in the human sense of the word
        #. Is a general class of algorithms that are able to approximate complicated,
           non-linear processe
#. 1.1 - Deep learning revolution
    a) Deep learning is a general class of algorithms that are able to approximate
       complicated, nonlinear processes very, very effectively,
    #) Most ML relied heavily on feature engineering.
        #. Transform the data to facilitate downstream algorithm
        #. Consider digit recognition ex
            * Maybe make a set of filters to estimate edge count and direction and 
              then predict digit.
            * Maybe consider holes, loops, etc
        #. Maybe adjust filters as training proceeds
    #) Deep Learning (DL) finds representations automatically from raw data
        #. Often better than the handcrafted data
        #. Requirements
            * Way to ingest data we have at hand
            * Define the DL machine
            * Automated way (i.e. training) to obtain useful representations and
              make machine produce desired outputs
    #) Fig 1.1 : (see \ref{fig1.1})
        #. Illustrates feature engineering vs. DL and its advantages
        #. Traditionally
            * data scientist hand-crafted engineered features
        #. Deep learning
            * feed in the raw data and it extracts hierarchical features automatically
        #. ![Fig 1.1 - handcrafted features vs increased data reqs\label{fig1.1}](figs/fig_1.1.png)
#. 1.2 - Pytorch for deep learning
#. 1.3 - Why PyTorch?
    a) Pythonic, ubiquitous, allows GPU use
    #) Flexible, allows complex implementation of ideas w/o unneccessary complexity
       from the library.
#. 1.3.1 - The Deep Learning Competitive landscape
    a) Before pytorch's first release
        #. Theono / TensorFlow (TF) were the premiere low-level libs with user
           defined computational graph
        #. Lasagne / Keras were high-level wrappers around Theono, TF and CNTK as well
        #. Caffe, CHainer, DyNet, Torch all had their own niches
    #) W/in 2 years, the community consolidated behind PyTorch or TF
        #. Theono ceased active dev
        #. TF
            * consumed Keras
            * Added Pytorch-like "eager mode"
        #. JAX, by google, became the Numpy-like equivalent for FPUs
        #. Pytorch
            * Consumed Caffe2 for backend
            * Replaced most low-level code reused from Lua-based Torch project
            * Added support for ONNX, vendor-neutral model description and exchange
              format
            * Added delayed-execution 'graph mode' runtime called TorchScript
            * Replaced CNTK and Chainer as framework of choice for respective
              corporate sponsors
    #) TF is more industry-wide community, PyTorch used by academia / teaching
#. 1.4 - Overview of how PyTorch supports deep learning projects
    a) 2 Core features
        #. Provides Tensor data structure
        #. Provides ability of tracking operations done on Tensors and track their 
           derivatives
            * 'autograd' engine under the hoold
    #) torch.nn
        #. Core of pytorch, provides
            * NN layers, fully connected layers, convolutional layers,
              activation functions and loss functions
        #. Still Needs -
            * training 
            * optimizer
    #) ![Fig 1.2\label{fig1.2}](figs/fig_1.2.png) : Basic, high-level structure of a pytorch project...
        #. From left to right...
        #. Dataset (torch.utils.data)
            * Bridge from Data Source -> Tensor
            * Can parallelize and assemble data into batches
        #. Training model
            * Inputs untrained model and batch tensors
            * Outputs trained model
            * Evaluates based off of loss function (provided in torch.nn)
            * optimizer (torch.optim) to change model weights after calculating
              loss function
            * Can use torch.nn.parallel.DistributedDataParallel and torch.distributed
        #. Trained model deployed in production
#. 1.5 - Hardware and software requirements
    a) MacOS binaries don't include anything CUDA enabled b/c macs don't have CUDA
       enabled GPUs


Chapter 2 : Pre-trained Methods
=============================================
1. Intro
#. 2.1 A pretrained network that recognizes the subject of an image
    a) Working with ImageNet and Wordnet
        #. http://imagenet.stanford.edu
        #. http://wordnet.princeton.edu
    #) ImageNet Large Scale Visual Recognition Challeng
        #. Competition started in 2010
        #. 1.2million images
    #) Each file is an RGB image w/ h x w and 3 color channels
    #) Output is a 1000 element tensor corresponding with the 1000 possible class 
       values
        #. See ![Fig 2.2 : The inference process \label{fig2.2}](figs/fig_2.2.png) 
#. 2.1.1 Obtaining a pretrained network for image recognition
    a) Download model
        #. [TOrch Vision](https://github.com/pytorch/vision)
        #. [AlexNet](http://mng.bz/lo6z)
        #. [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
        #. [Inception 3](https://arxiv.org/pdf/1512.00567.pdf)
    #) AlexNet
        #. Won ILSVRC in 2012 w/ an error rate of 15.4%, 2nd place was 26.2%
    #) Fig 2.3 : The AlexNet architecture
        #. 



1. A pretrained network that recognizes the subject of an
image 17
Obtaining a pretrained network for image recognition 19
AlexNet 20 ResNet 22 Ready, set, almost run 22
Run! 25
#. A pretrained model that fakes it until it makes it 27
The GAN game 28 CycleGAN 29 A network that turns
horses into zebras 30
#. A pretrained network that describes scenes 33
#. Torch Hub


Chapter 3 : It starts with a tensor
=============================================
1. The world as floating-point numbers 40
#. Tensors: Multidimensional arrays 42
From Python lists to PyTorch tensors tensors 43 The essence of tensors 43
#. Indexing tensors 46
#. Named tensors 46
42 Constructing our first
#. Tensor element types 50
Specifying the numeric type with dtype 50 A dtype for every
occasion 51 Managing a tensor’s dtype attribute 51
#. The tensor API 52
#. Tensors: Scenic views of storage 53
Indexing into storage 54 Modifying stored values: In-place
operations 55
#. Tensor metadata: Size, offset, and stride 55
Views of another tensor’s storage 56 Transposing without
copying 58 Transposing in higher dimensions 60
Contiguous tensors 60
#. Moving tensors to the GPU 62
Managing a tensor’s device attribute 63
#. NumPy interoperability 64
#. Generalized tensors are tensors, too 65
#. Serializing tensors 66
Serializing to HDF5 with h5py 67


Chapter 4 : Real-world data representation using tensors
=============================================
1. Working with images 71
Adding color channels 72 Loading an image file 72
Changing the layout 73 Normalizing the data 74
#. 3D images: Volumetric data 75
Loading a specialized format 76
#. Representing tabular data 77
Using a real-world dataset 77 Loading a wine data tensor 78
Representing scores 81 One-hot encoding 81 When to
categorize 83 Finding thresholds 84
#. Working with time series 87
Adding a time dimension 88 Shaping the data by time
period 89 Ready for training 90
#. Representing text 93
Converting text to numbers 94 One-hot-encoding characters 94


Chapter 5 : The mechanics of learning
=============================================
The mechanics of learning 103
1. A timeless lesson in modeling 104
#. Learning is just parameter estimation 106
A hot problem 107 Gathering some data 107 Visualizing
the data 108 Choosing a linear model as a first try 108
#. Less loss is what we want 109
#. Down along the gradient 113
Decreasing loss 113 Getting analytical 114 Iterating to fit
the model 116 Normalizing inputs 119 Visualizing
(again) 122
#. PyTorch’s autograd: Backpropagating all things 123
Computing the gradient automatically 123 Optimizers a la


Chapter 6 : Using a nerual network to fit the data 
=============================================
1. Artificial neurons 142
Composing a multilayer network 144 Understanding the error
function 144 All we need is activation 145 More activation
functions 147 Choosing the best activation function 148
What learning means for a neural network 149
#. The PyTorch nn module 151
Using __call__ rather than forward 152 Returning to the linear
model 153
#. Finally a neural network 158
Replacing the linear model 158 Inspecting the parameters 159


Chapter 7 : Telling birds from airplanes : Learning from images
=============================================
1. A dataset of tiny images 165
Downloading CIFAR-10 166 The Dataset class 166
Dataset transforms 168 Normalizing data 170
#. Distinguishing birds from airplanes 172
Building the dataset 173 A fully connected model 174
Output of a classifier 175 Representing the output as
probabilities 176 A loss for classifying 180 Training the


Chapter 8 : Using convolutions to generalize
=============================================
1. The case for convolutions 194
What convolutions do 194
#. Convolutions in action 196
Padding the boundary 198 Detecting features with
convolutions 200 Looking further with depth and pooling 202
Putting it all together for our network 205
#. Subclassing nn.Module 207
Our network as an nn.Module 208 How PyTorch keeps track of
parameters and submodules 209 The functional API 210
#. Training our convnet 212
Measuring accuracy 214 Saving and loading our model 214
Training on the GPU 215
#. Model design 217
Adding memory capacity: Width 218 Helping our model to
converge and generalize: Regularization 219 Going deeper to
learn more complex structures: Depth 223 Comparing the designs


Chapter 9 : Using PyTorch to fight cancer
=============================================
1. Introduction to the use case 236
#. Preparing for a large-scale project 237
#. What is a CT scan, exactly? 238
#. The project: An end-to-end detector for lung cancer Why can’t we just throw data at a neural network until it
works? 245 What is a nodule? 249 Our data source:
The LUNA Grand Challenge 251 Downloading the LUNA


Chapter 10 : Combining data sources into a unified dataset
=============================================
1. Raw CT data files 256
#. Parsing LUNA’s annotation data 256
Training and validation sets 258 Unifying our annotation and
candidate data 259
#. Loading individual CT scans 262
Hounsfield Units 264
#. Locating a nodule using the patient coordinate system The patient coordinate system 265 CT scan shape and
voxel sizes 267 Converting between millimeters and voxel
addresses 268 Extracting a nodule from a CT scan 270
#. A straightforward dataset implementation 271
Caching candidate arrays with the getCtRawCandidate
function 274 Constructing our dataset in LunaDataset
.__init__ 275 A training/validation split 275 Rendering


Chapter 11 : Training a classificaiton model to detect suspected tumors
=============================================
1. Training a classification model to detect suspected tumors 11.1 A foundational model and training loop 280
#. The main entry point for our application 282
#. Pretraining setup and initialization 284
Initializing the model and optimizer 285 Care and feeding of
data loaders 287
#. Our first-pass neural network design 289
The core convolutions 290 The full model 293
#. Training and validating the model 295
The computeBatchLoss function 297 The validation loop is
similar 299
#. Outputting performance metrics 300
The logMetrics function 301
#. Running the training script 304
Needed data for training 305 Interlude: The
enumerateWithEstimate function 306
#. Evaluating the model: Getting 99.7% correct means we’re
done, right? 308
#. Graphing training metrics with TensorBoard 309
Running TensorBoard 309 Adding TensorBoard support to the
metrics logging function 313
#. Why isn’t the model learning to detect nodules? 315


Chapter 12 : Improving training with metrics and augmentation 
=============================================
Improving training with metrics and augmentation 318
1. High-level plan for improvement 319
#. Good dogs vs. bad guys: False positives and false negatives 12.3 Graphing the positives and negatives 322
Recall is Roxie’s strength 324 Precision is Preston’s forte 326
Implementing precision and recall in logMetrics 327 Our
ultimate performance metric: The F1 score 328 How does our
model perform with our new metrics? 332
#. What does an ideal dataset look like? 334
Making the data look less like the actual and more like the “ideal” Contrasting training with a balanced LunaDataset to previous
runs 341 Recognizing the symptoms of overfitting 343
#. Revisiting the problem of overfitting 345
An overfit face-to-age prediction model 345
#. Preventing overfitting with data augmentation 346
Specific data augmentation techniques 347■ Seeing the
improvement from data augmentation 352


Chapter 13 : Using segmentation to find suspected nodules
=============================================
13 Using segmentation to find suspected nodules 357
1. Adding a second model to our project 358
#. Various types of segmentation 360
#. Semantic segmentation: Per-pixel classification 361
The U-Net architecture 364
#. Updating the model for segmentation 366
Adapting an off-the-shelf model to our project 367
#. Updating the dataset for segmentation 369
U-Net has very specific input size requirements 370 U-Net trade-
offs for 3D vs. 2D data 370 Building the ground truth
data 371 Implementing Luna2dSegmentationDataset 378
Designing our training and validation data 382 Implementing
TrainingLuna2dSegmentationDataset 383 Augmenting on the
GPU 384
#. Updating the training script for segmentation 386
Initializing our segmentation and augmentation models 387
Using the Adam optimizer 388 Dice loss 389 Getting images
into TensorBoard 392 Updating our metrics logging 396
Saving our model 397
#. Results 399


Chapter 14 : End-to-end nodule analysis and where to go next 
=============================================
1. Towards the finish line 405
#. Independence of the validation set 407
#. Bridging CT segmentation and nodule candidate
classification 408
Segmentation 410 Grouping voxels into nodule candidates 411
Did we find a nodule? Classification to reduce false positives 412
#. Quantitative validation 416
#. Predicting malignancy 417
Getting malignancy information 417 An area under the curve
baseline: Classifying by diameter 419 Reusing preexisting
weights: Fine-tuning 422 More output in TensorBoard 428
#. What we see when we diagnose 432
Training, validation, and test sets 433
#. What next? Additional sources of inspiration (and data) Preventing overfitting: Better regularization 434 Refined training
data 437 Competition results and research papers 438


Chapter 15 : Deploying to production 
=============================================
15 Deploying to production 445
1. Serving PyTorch models 446
Our model behind a Flask server 446 What we want from
deployment 448 Request batching 449
#. Exporting models 455
Interoperability beyond PyTorch with ONNX 455 PyTorch’s own
export: Tracing 456 Our server with a traced model 458
#. Interacting with the PyTorch JIT 458
What to expect from moving beyond classic Python/PyTorch 458
The dual nature of PyTorch as interface and backend 460
TorchScript 461 Scripting the gaps of traceability 464
#. LibTorch: PyTorch in C++ 465
Running JITed models from C++ 465 C++ from the start: The
C++ API 468
#. Going mobile 472
Improving efficiency: Model design and quantization 475
#. Emerging technology: Enterprise serving of PyTorch
models 476

