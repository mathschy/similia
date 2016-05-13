// This file defines a simple similia client: it reads a query image from the
// command line, and sends the request to a running similia server using gRPC.
// The main is to be able to stress test and experiment with the similia
// server.
//
// Sample usage:
// ./build/similia_test_client --logtostderr --image path/to/queryimage.jpg
#include <string>

#include <boost/filesystem/path.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <grpc++/grpc++.h>
#include <similia/proto/similia.pb.h>
#include <similia/proto/similia.grpc.pb.h>

#include <similia/common/utils/file_utils.h>
#include <similia/utils/cropping_utils.h>
#include <similia/utils/features_extractor.h>


using boost::filesystem::path;
using google::protobuf::RepeatedField;

using similia::proto::Similia;
using similia::proto::SimiliaSearchRequest;
using similia::proto::SimiliaSearchResponse;

DEFINE_string(image,
              "",
              "Path to the .jpg image to search for nearest neighbors.");
DEFINE_string(weights,
              "similia/data/bvlc_googlenet.caffemodel",
              "Path to the .caffemodel file.");
DEFINE_string(deploy_prototxt,
              "similia/data/deploy_mc.prototxt",
              "Path to the deploy .prototxt file.");
DEFINE_string(blob_names, "pool5/7x7_s1", "Name of the blob to extract features from.");
DEFINE_int32(gpu, -1, "Run in GPU mode on given device ID. -1 means run on CPU.");
DEFINE_string(similia_server_host, "localhost", "Host of similia_server");
DEFINE_int32(similia_server_port, 20001, "Port of similia_server");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  // Compute features for the image.
  similia::FeaturesExtractor fe(FLAGS_weights, FLAGS_deploy_prototxt, FLAGS_blob_names,
                                FLAGS_gpu);
  const path input_path(FLAGS_image);
  LOG(INFO) << "adding image: " << input_path.string();
  std::vector<float> features_vec = fe.CropAndExtractFeatures(common_utils::ReadFromFileOrDie(input_path));
  CHECK(!features_vec.empty());

  // Init connection to server
  const std::string similia_server_address = FLAGS_similia_server_host + ":" +
      std::to_string(FLAGS_similia_server_port);
  LOG(INFO) << "connection to similia: " << similia_server_address;
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(similia_server_address,
                                                               grpc::InsecureChannelCredentials());
  std::unique_ptr<Similia::Stub> client = Similia::NewStub(channel);

  // Prepare the request
  SimiliaSearchRequest request;
  request.set_image_id(input_path.filename().stem().string());
  request.set_num_nearest(10);
  *request.mutable_features()->mutable_element() = RepeatedField<float>(features_vec.begin(), features_vec.end());

  LOG(INFO) << "sending request...";
  SimiliaSearchResponse response;
  grpc::ClientContext context;
  grpc::Status status = client->SimiliaSearch(&context, request, &response);
  if (status.ok()) {
    std::string out;
    google::protobuf::TextFormat::PrintToString(response, &out);
    LOG(INFO) << "Request success: " << out;
    return 0;
  } else {
    LOG(ERROR) << "request failed! " << status.error_code() << ": " << status.error_message();
    return 1;
  }

}

