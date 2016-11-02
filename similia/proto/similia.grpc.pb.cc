// Generated by the gRPC protobuf plugin.
// If you make any local change, they will be lost.
// source: similia/proto/similia.proto

#include "similia/proto/similia.pb.h"
#include "similia/proto/similia.grpc.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/channel_interface.h>
#include <grpc++/impl/codegen/client_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/rpc_service_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/sync_stream.h>
namespace similia {
namespace proto {

static const char* Similia_method_names[] = {
  "/similia.proto.Similia/SimiliaSearch",
};

std::unique_ptr< Similia::Stub> Similia::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  std::unique_ptr< Similia::Stub> stub(new Similia::Stub(channel));
  return stub;
}

Similia::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_SimiliaSearch_(Similia_method_names[0], ::grpc::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status Similia::Stub::SimiliaSearch(::grpc::ClientContext* context, const ::similia::proto::SimiliaSearchRequest& request, ::similia::proto::SimiliaSearchResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_SimiliaSearch_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::SimiliaSearchResponse>* Similia::Stub::AsyncSimiliaSearchRaw(::grpc::ClientContext* context, const ::similia::proto::SimiliaSearchRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::SimiliaSearchResponse>(channel_.get(), cq, rpcmethod_SimiliaSearch_, context, request);
}

Similia::Service::Service() {
  (void)Similia_method_names;
  AddMethod(new ::grpc::RpcServiceMethod(
      Similia_method_names[0],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< Similia::Service, ::similia::proto::SimiliaSearchRequest, ::similia::proto::SimiliaSearchResponse>(
          std::mem_fn(&Similia::Service::SimiliaSearch), this)));
}

Similia::Service::~Service() {
}

::grpc::Status Similia::Service::SimiliaSearch(::grpc::ServerContext* context, const ::similia::proto::SimiliaSearchRequest* request, ::similia::proto::SimiliaSearchResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


static const char* InvertedMultiIndex_method_names[] = {
  "/similia.proto.InvertedMultiIndex/Add",
  "/similia.proto.InvertedMultiIndex/Get",
  "/similia.proto.InvertedMultiIndex/Delete",
  "/similia.proto.InvertedMultiIndex/MultiGet",
  "/similia.proto.InvertedMultiIndex/MultiCount",
  "/similia.proto.InvertedMultiIndex/MultiAdd",
};

std::unique_ptr< InvertedMultiIndex::Stub> InvertedMultiIndex::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  std::unique_ptr< InvertedMultiIndex::Stub> stub(new InvertedMultiIndex::Stub(channel));
  return stub;
}

InvertedMultiIndex::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_Add_(InvertedMultiIndex_method_names[0], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_Get_(InvertedMultiIndex_method_names[1], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_Delete_(InvertedMultiIndex_method_names[2], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_MultiGet_(InvertedMultiIndex_method_names[3], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_MultiCount_(InvertedMultiIndex_method_names[4], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_MultiAdd_(InvertedMultiIndex_method_names[5], ::grpc::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status InvertedMultiIndex::Stub::Add(::grpc::ClientContext* context, const ::similia::proto::MultiIndexAddRequest& request, ::similia::proto::MultiIndexAddResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_Add_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexAddResponse>* InvertedMultiIndex::Stub::AsyncAddRaw(::grpc::ClientContext* context, const ::similia::proto::MultiIndexAddRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexAddResponse>(channel_.get(), cq, rpcmethod_Add_, context, request);
}

::grpc::Status InvertedMultiIndex::Stub::Get(::grpc::ClientContext* context, const ::similia::proto::MultiIndexGetRequest& request, ::similia::proto::MultiIndexGetResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_Get_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexGetResponse>* InvertedMultiIndex::Stub::AsyncGetRaw(::grpc::ClientContext* context, const ::similia::proto::MultiIndexGetRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexGetResponse>(channel_.get(), cq, rpcmethod_Get_, context, request);
}

::grpc::Status InvertedMultiIndex::Stub::Delete(::grpc::ClientContext* context, const ::similia::proto::MultiIndexDeleteRequest& request, ::similia::proto::MultiIndexDeleteResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_Delete_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexDeleteResponse>* InvertedMultiIndex::Stub::AsyncDeleteRaw(::grpc::ClientContext* context, const ::similia::proto::MultiIndexDeleteRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexDeleteResponse>(channel_.get(), cq, rpcmethod_Delete_, context, request);
}

::grpc::Status InvertedMultiIndex::Stub::MultiGet(::grpc::ClientContext* context, const ::similia::proto::MultiIndexMultiGetRequest& request, ::similia::proto::MultiIndexMultiGetResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_MultiGet_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexMultiGetResponse>* InvertedMultiIndex::Stub::AsyncMultiGetRaw(::grpc::ClientContext* context, const ::similia::proto::MultiIndexMultiGetRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexMultiGetResponse>(channel_.get(), cq, rpcmethod_MultiGet_, context, request);
}

::grpc::Status InvertedMultiIndex::Stub::MultiCount(::grpc::ClientContext* context, const ::similia::proto::MultiIndexMultiCountRequest& request, ::similia::proto::MultiIndexMultiCountResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_MultiCount_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexMultiCountResponse>* InvertedMultiIndex::Stub::AsyncMultiCountRaw(::grpc::ClientContext* context, const ::similia::proto::MultiIndexMultiCountRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexMultiCountResponse>(channel_.get(), cq, rpcmethod_MultiCount_, context, request);
}

::grpc::Status InvertedMultiIndex::Stub::MultiAdd(::grpc::ClientContext* context, const ::similia::proto::MultiIndexMultiAddRequest& request, ::similia::proto::MultiIndexMultiAddResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_MultiAdd_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexMultiAddResponse>* InvertedMultiIndex::Stub::AsyncMultiAddRaw(::grpc::ClientContext* context, const ::similia::proto::MultiIndexMultiAddRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::MultiIndexMultiAddResponse>(channel_.get(), cq, rpcmethod_MultiAdd_, context, request);
}

InvertedMultiIndex::Service::Service() {
  (void)InvertedMultiIndex_method_names;
  AddMethod(new ::grpc::RpcServiceMethod(
      InvertedMultiIndex_method_names[0],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< InvertedMultiIndex::Service, ::similia::proto::MultiIndexAddRequest, ::similia::proto::MultiIndexAddResponse>(
          std::mem_fn(&InvertedMultiIndex::Service::Add), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      InvertedMultiIndex_method_names[1],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< InvertedMultiIndex::Service, ::similia::proto::MultiIndexGetRequest, ::similia::proto::MultiIndexGetResponse>(
          std::mem_fn(&InvertedMultiIndex::Service::Get), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      InvertedMultiIndex_method_names[2],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< InvertedMultiIndex::Service, ::similia::proto::MultiIndexDeleteRequest, ::similia::proto::MultiIndexDeleteResponse>(
          std::mem_fn(&InvertedMultiIndex::Service::Delete), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      InvertedMultiIndex_method_names[3],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< InvertedMultiIndex::Service, ::similia::proto::MultiIndexMultiGetRequest, ::similia::proto::MultiIndexMultiGetResponse>(
          std::mem_fn(&InvertedMultiIndex::Service::MultiGet), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      InvertedMultiIndex_method_names[4],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< InvertedMultiIndex::Service, ::similia::proto::MultiIndexMultiCountRequest, ::similia::proto::MultiIndexMultiCountResponse>(
          std::mem_fn(&InvertedMultiIndex::Service::MultiCount), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      InvertedMultiIndex_method_names[5],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< InvertedMultiIndex::Service, ::similia::proto::MultiIndexMultiAddRequest, ::similia::proto::MultiIndexMultiAddResponse>(
          std::mem_fn(&InvertedMultiIndex::Service::MultiAdd), this)));
}

InvertedMultiIndex::Service::~Service() {
}

::grpc::Status InvertedMultiIndex::Service::Add(::grpc::ServerContext* context, const ::similia::proto::MultiIndexAddRequest* request, ::similia::proto::MultiIndexAddResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status InvertedMultiIndex::Service::Get(::grpc::ServerContext* context, const ::similia::proto::MultiIndexGetRequest* request, ::similia::proto::MultiIndexGetResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status InvertedMultiIndex::Service::Delete(::grpc::ServerContext* context, const ::similia::proto::MultiIndexDeleteRequest* request, ::similia::proto::MultiIndexDeleteResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status InvertedMultiIndex::Service::MultiGet(::grpc::ServerContext* context, const ::similia::proto::MultiIndexMultiGetRequest* request, ::similia::proto::MultiIndexMultiGetResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status InvertedMultiIndex::Service::MultiCount(::grpc::ServerContext* context, const ::similia::proto::MultiIndexMultiCountRequest* request, ::similia::proto::MultiIndexMultiCountResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status InvertedMultiIndex::Service::MultiAdd(::grpc::ServerContext* context, const ::similia::proto::MultiIndexMultiAddRequest* request, ::similia::proto::MultiIndexMultiAddResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


static const char* MultiProductQuantizer_method_names[] = {
  "/similia.proto.MultiProductQuantizer/Quantize",
};

std::unique_ptr< MultiProductQuantizer::Stub> MultiProductQuantizer::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  std::unique_ptr< MultiProductQuantizer::Stub> stub(new MultiProductQuantizer::Stub(channel));
  return stub;
}

MultiProductQuantizer::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_Quantize_(MultiProductQuantizer_method_names[0], ::grpc::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status MultiProductQuantizer::Stub::Quantize(::grpc::ClientContext* context, const ::similia::proto::QuantizationRequest& request, ::similia::proto::QuantizationResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_Quantize_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::similia::proto::QuantizationResponse>* MultiProductQuantizer::Stub::AsyncQuantizeRaw(::grpc::ClientContext* context, const ::similia::proto::QuantizationRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::similia::proto::QuantizationResponse>(channel_.get(), cq, rpcmethod_Quantize_, context, request);
}

MultiProductQuantizer::Service::Service() {
  (void)MultiProductQuantizer_method_names;
  AddMethod(new ::grpc::RpcServiceMethod(
      MultiProductQuantizer_method_names[0],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< MultiProductQuantizer::Service, ::similia::proto::QuantizationRequest, ::similia::proto::QuantizationResponse>(
          std::mem_fn(&MultiProductQuantizer::Service::Quantize), this)));
}

MultiProductQuantizer::Service::~Service() {
}

::grpc::Status MultiProductQuantizer::Service::Quantize(::grpc::ServerContext* context, const ::similia::proto::QuantizationRequest* request, ::similia::proto::QuantizationResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace similia
}  // namespace proto

