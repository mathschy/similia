// This file defines a simple inverted_multi_index client: it reads a cluster key from the
// command line, and sends the request to a running inverted multi index server using gRPC.
//
// Sample usage:
//
//     ./build/inverted_multi_index_test_client --logtostderr --key 275,8133
#include <string>

#include <boost/algorithm/string.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <grpc++/grpc++.h>

#include <similia/proto/compressed_elements_generated.h>
#include <similia/proto/similia.pb.h>
#include <similia/proto/similia.grpc.pb.h>

namespace fbs = similia::fbs;
using similia::proto::IndexingClustersIds;
using similia::proto::InvertedMultiIndex;
using similia::proto::MultiIndexGetRequest;
using similia::proto::MultiIndexGetResponse;

DEFINE_string(key, "", "Cluster key of the form: cluster_id1,cluster_id2");
DEFINE_string(inverted_multi_index_server_host, "localhost", "Host of the inverted_multi_index_server.");
DEFINE_int32(inverted_multi_index_server_port, 20000, "Port of the inverted_multi_index_server.");


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  std::string cluster_key = FLAGS_key;
  CHECK(!cluster_key.empty()) << "please specify a cluster key";

  // Init connection to server
  const std::string address = FLAGS_inverted_multi_index_server_host + ":" +
      std::to_string(FLAGS_inverted_multi_index_server_port);
  LOG(INFO) << "connection to inverted_multi_index_server: " << address;
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  std::unique_ptr<InvertedMultiIndex::Stub> client = InvertedMultiIndex::NewStub(channel);

  // parse cluster key
  std::vector<std::string> tokens;
  boost::split(tokens, FLAGS_key, boost::is_any_of(","));
  CHECK_EQ(tokens.size(), 2);
  int cluster_id1 = std::stoi(tokens[0]);
  int cluster_id2 = std::stoi(tokens[1]);

  // Prepare the request
  MultiIndexGetRequest request;
  IndexingClustersIds indexing_clusters_ids;
  indexing_clusters_ids.add_id(cluster_id1);
  indexing_clusters_ids.add_id(cluster_id2);
  *request.mutable_indexing_ids() = indexing_clusters_ids;

  LOG(INFO) << "sending request...";
  MultiIndexGetResponse response;
  grpc::ClientContext context;
  grpc::Status status = client->Get(&context, request, &response);
  if (status.ok()) {
    std::string out;
    google::protobuf::TextFormat::Printer printer;
    printer.SetUseShortRepeatedPrimitives(true);
    printer.PrintToString(response, &out);
    LOG(INFO) << "Request success: " << out;
    const fbs::CompressedElements* compressed_elements = fbs::GetCompressedElements(
        reinterpret_cast<const uint8_t*>(response.compressed_elements().data()));
    LOG(INFO) << "There are " << compressed_elements->id()->size() << " elements in cluster " << FLAGS_key;
    CHECK_EQ(compressed_elements->id()->size(), compressed_elements->compressing_ids()->size());
    for (int i = 0; i < compressed_elements->id()->size(); ++i) {
      std::string line;
      line += "id: " + std::string(compressed_elements->id()->Get(i)->data()) + " , compressing_ids: [";
      for (int j = 0; j < compressed_elements->compressing_ids()->Get(i)->compressing_ids()->size(); ++j) {
        line += std::to_string(compressed_elements->compressing_ids()->Get(i)->compressing_ids()->Get(j)) + ", ";
      }
      line.pop_back();
      line[line.size()-1] = ']';
      LOG(INFO) << line;
    }
    return 0;
  } else {
    LOG(ERROR) << "request failed! " << status.error_message();
    return 1;
  }
}
