#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc++/grpc++.h>

#include <similia/servers/similia_service.h>
#include <similia/utils/candidates_finder.h>
#include <similia/utils/candidates_reranker.h>
#include <similia/utils/features_library.h>


DEFINE_string(root_path_to_indexing_clusters_features, "similia/data/indexing_clusters_features_",
              "Path to the root file of indexing clusters features. add 0.csv and 1.csv to get the actual files.");
DEFINE_string(root_path_to_compressing_clusters_features, "similia/data/compressing_clusters_features_",
              "Path to the root file of compressing clusters features.");
DEFINE_string(path_to_indexing_rotation_matrix, "similia/data/indexing_rotation_matrix.pb",
              "Path to the file that contains the indexing-level rotation matrix protobuf message.");
DEFINE_string(path_to_compressing_rotation_matrix, "similia/data/compressing_rotation_matrix.pb",
              "Path to the file that contains the compressing-level rotation matrix protobuf message.");
DEFINE_string(inverted_multi_index_server_host, "localhost", "host of the inverted_multi_index_server to connect to.");
DEFINE_int32(inverted_multi_index_server_port, 20000, "port of the inverted_multi_index_server to connect to.");
DEFINE_int32(port, 20001, "port for the similia_server to listen to.");

using similia::proto::InvertedMultiIndex;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  // configure connection to inverted_multi_index_server
  const std::string inverted_multi_index_server_address = FLAGS_inverted_multi_index_server_host + ":" +
      std::to_string(FLAGS_inverted_multi_index_server_port);
  LOG(INFO) << "connection to inverted_multi_index: " << inverted_multi_index_server_address;
  std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(inverted_multi_index_server_address,
                                                               grpc::InsecureChannelCredentials());
  std::unique_ptr<InvertedMultiIndex::Stub> inverted_multi_index_client = InvertedMultiIndex::NewStub(channel);

  // initialize classes
  LOG(INFO) << "Initializing CandidatesFinder...";
  similia::CandidatesFinder cf(FLAGS_root_path_to_indexing_clusters_features + "0.csv",
                               FLAGS_root_path_to_indexing_clusters_features + "1.csv",
                               std::move(inverted_multi_index_client));
  LOG(INFO) << "Initialized CandidatesFinder. Initializing CandidatesReranker...";
  similia::CandidatesReranker cr(FLAGS_root_path_to_indexing_clusters_features,
                                 FLAGS_root_path_to_compressing_clusters_features,
                                 FLAGS_path_to_indexing_rotation_matrix,
                                 FLAGS_path_to_compressing_rotation_matrix);
  LOG(INFO) << "Initialized CandidatesReranker. Initializing SimiliaService...";
  similia::SimiliaService service(&cf, &cr, FLAGS_path_to_indexing_rotation_matrix,
                                  FLAGS_path_to_compressing_rotation_matrix);
  LOG(INFO) << "Initialized SimiliaService.";

  // launch grpc service
  const std::string address = "0.0.0.0:" + std::to_string(FLAGS_port);
  std::unique_ptr<grpc::ServerBuilder> builder(new grpc::ServerBuilder());
  builder->AddListeningPort(address, grpc::InsecureServerCredentials());
  builder->RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder->BuildAndStart());
  LOG(INFO) << "Similia server is taking the stage on: " << address;
  server->Wait();
  return 0;
}
