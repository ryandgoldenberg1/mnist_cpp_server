# C++ MNIST Server

A minimal example for training your PyTorch model in Python, and serving in C++. It trains a convolutional neural network using the MNIST dataset in PyTorch, persists it using TorchScript, and serves it in a C++ webserver.


## How To Run

### 1. Install the Dependencies

```bash
./bin/install_deps.sh # (on Amazon Linux 2)
```

The project has the following dependencies which need to be installed:
* Python 3.6
* [PyTorch 1.6](https://pytorch.org/)
* [Matplotlib 3.3](https://matplotlib.org/)
* [CMake 3.16](https://cmake.org/)
* [nlohman/json 3.2](https://github.com/nlohmann/json)
* [cpp-httplib 0.7](https://github.com/yhirose/cpp-httplib)
* [LibTorch 1.6](https://pytorch.org/)

To install them, you can use the `bin/install_deps.sh` script if you are on Amazon Linux or a compatible OS. Otherwise you will need to adapt the commands there to add these packages.

### 2. Build the Server Executable

```bash
./bin/build.sh
```

This will compile and link the C++ code, and place an executable `build/server` which can be used to start the model server.

### 3. Train the Model

```bash
python3 train.py --dataset_root data --download
```

This will train a convolutional neural network for a few epochs and output the TorchScript representation to `model.pt`.

### 4. Start the Webserver

```bash
./build/server
```

This will start the C++ webserver on port 3000. You can check that it's working as expected by opening another terminal and running

```bash
curl http://localhost:3000/predict -XPOST -d @data/example_1.json
```

which should return 0.

### 5. Make the Predictions

```
python3 predict.py
```

You must have the server from step 4 running locally. This script will make requests with each of the examples in
`data/example_{1..5}.json`, similar to the `curl` command above, and display the response together with the image.


## References

Code in this repo is drawn from
* [PyTorch: LOADING A TORCHSCRIPT MODEL IN C++](https://pytorch.org/tutorials/advanced/cpp_export.html)
* [PyTorch MNIST](https://github.com/pytorch/examples/tree/master/mnist)
