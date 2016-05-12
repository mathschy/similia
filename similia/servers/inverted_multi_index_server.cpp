#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpc++/grpc++.h>

#include <similia/servers/inverted_multi_index_service.h>
#include <similia/utils/inverted_multi_index.h>


DEFINE_string(db_path, "test", "path to a folder that will contain the rocksdb files.");
DEFINE_int32(port, 20000, "Port for the server to listen to.");

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  LOG(INFO) << "Initializing InvertedMultiIndex...";
  similia::InvertedMultiIndex inverted_multi_index(FLAGS_db_path);
  LOG(INFO) << "Initialized InvertedMultiIndex. Initializing InvertedMultiIndexService...";
  similia::InvertedMultiIndexService service(&inverted_multi_index);
  LOG(INFO) << "Initialized InvertedMultiIndexService.";

  const std::string address = "0.0.0.0:" + std::to_string(FLAGS_port);
  std::unique_ptr<grpc::ServerBuilder> builder(new grpc::ServerBuilder());
  builder->AddListeningPort(address, grpc::InsecureServerCredentials());
  builder->RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder->BuildAndStart());
  LOG(INFO) << "InvertedMultiIndex server is taking the stage on: " << address;
  server->Wait();
  return 0;
}