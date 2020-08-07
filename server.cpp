#include <iostream>
#include <sstream>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <torch/script.h>
using json = nlohmann::json;


int main(int argc, char** argv) {
    std::string model_path {"model.pt"};
    if (argc > 1) {
        model_path = argv[1];
    }

    torch::jit::script::Module model = torch::jit::load(model_path);
    httplib::Server server;

    server.Post("/predict", [&](const httplib::Request &request, httplib::Response &response) {
        // Parse the request body into a torch::Tensor object
        // Expects the body to be a 28x28 array of doubles in json format
        auto body = json::parse(request.body);
        auto vector = body.get<std::vector<std::vector<double>>>();
        auto tensor = torch::zeros({1, 1, 28, 28}, torch::TensorOptions().dtype(torch::kFloat32));
        for (std::size_t i {0}; i < 28; i++) {
            for (std::size_t j {0}; j < 28; j++) {
                tensor[0][0][i][j] = vector.at(i).at(j);
            }
        }

        // Make the prediction using our model
        auto logits = model.forward(std::vector<torch::jit::IValue> {tensor}).toTensor();
        auto pred = logits.argmax(-1).item().to<int>();

        // Output the result as a string
        std::ostringstream content;
        content << pred;
        response.set_content(content.str(), "text/plain");
    });

    std::cout << "Starting server on http://localhost:3000/" << std::endl;
    server.listen("localhost", 3000);
    return 0;
}
