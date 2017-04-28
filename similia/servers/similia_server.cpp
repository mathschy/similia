#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc++/grpc++.h>

#include "similia/servers/similia_service.h"
#include "similia/utils/candidates_finder.h"
#include "similia/utils/candidates_reranker.h"
#include "similia/utils/features_library.h"


DEFINE_string(icf_root, "similia/data/indexing_clusters_features_",
              "Path to the root file of indexing clusters features. add 0.csv and 1.csv to get the actual files.");
DEFINE_string(ccf_root, "similia/data/compressing_clusters_features_",
              "Path to the root file of compressing clusters features.");
DEFINE_string(irm, "similia/data/indexing_rotation_matrix.pb",
              "Path to the file that contains the indexing-level rotation matrix protobuf message.");
DEFINE_string(crm, "similia/data/compressing_rotation_matrix.pb",
              "Path to the file that contains the compressing-level rotation matrix protobuf message.");
DEFINE_string(imis_host, "localhost", "host of the inverted_multi_index_server to connect to.");
DEFINE_int32(imis_port, 20000, "port of the inverted_multi_index_server to connect to.");
DEFINE_int32(port, 20001, "port for the similia_server to listen to.");

using similia::proto::InvertedMultiIndex;

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  // configure connection to inverted_multi_index_server
  const std::string inverted_multi_index_server_address = FLAGS_imis_host + ":" +
      std::to_string(FLAGS_imis_port);
  LOG(INFO) << "connection to inverted_multi_index: " << inverted_multi_index_server_address;
  grpc::ChannelArguments channel_arguments;
  // Allow this channel to receive 100MB message max.
  channel_arguments.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 100 * 1024 * 1024);
  std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
      inverted_multi_index_server_address,
      grpc::InsecureChannelCredentials(),
      channel_arguments);
  std::unique_ptr<InvertedMultiIndex::Stub> inverted_multi_index_client = InvertedMultiIndex::NewStub(channel);

  // initialize classes
  LOG(INFO) << "Initializing CandidatesFinder...";
  similia::CandidatesFinder cf(FLAGS_icf_root + "0.csv",
                               FLAGS_icf_root + "1.csv",
                               std::move(inverted_multi_index_client));
  LOG(INFO) << "Initialized CandidatesFinder. Initializing CandidatesReranker...";
  similia::CandidatesReranker cr(FLAGS_icf_root,
                                 FLAGS_ccf_root,
                                 FLAGS_irm,
                                 FLAGS_crm);
  LOG(INFO) << "Initialized CandidatesReranker. Initializing SimiliaService...";
  similia::SimiliaService service(&cf, &cr, FLAGS_irm,
                                  FLAGS_crm);
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
