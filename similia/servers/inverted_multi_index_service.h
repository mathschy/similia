#pragma once

#include "similia/proto/similia.grpc.pb.h"
#include "similia/utils/inverted_multi_index.h"

namespace similia {

class InvertedMultiIndexService final: public proto::InvertedMultiIndex::Service {
 public:
  explicit InvertedMultiIndexService(InvertedMultiIndex* inverted_multi_index)
      : inverted_multi_index_(inverted_multi_index) { }

  grpc::Status Add(grpc::ServerContext* context,
                   const proto::MultiIndexAddRequest* request,
                   proto::MultiIndexAddResponse* response) override;

  grpc::Status Get(grpc::ServerContext* context,
                   const proto::MultiIndexGetRequest* request,
                   proto::MultiIndexGetResponse* response) override;

  grpc::Status Delete(grpc::ServerContext* context,
                      const proto::MultiIndexDeleteRequest* request,
                      proto::MultiIndexDeleteResponse* response) override;

  grpc::Status MultiGet(grpc::ServerContext* context,
                        const proto::MultiIndexMultiGetRequest* request,
                        proto::MultiIndexMultiGetResponse* response) override;

  grpc::Status MultiCount(grpc::ServerContext* context,
                          const proto::MultiIndexMultiCountRequest* request,
                          proto::MultiIndexMultiCountResponse* response) override;

  grpc::Status MultiAdd(grpc::ServerContext* context,
                        const proto::MultiIndexMultiAddRequest* request,
                        proto::MultiIndexMultiAddResponse* response) override;

 private:
  InvertedMultiIndex* inverted_multi_index_;  // not owned.
};
}  // namespace similia
