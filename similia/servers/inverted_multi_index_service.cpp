#include <similia/servers/inverted_multi_index_service.h>

#include <chrono>

#include <glog/logging.h>

#include <similia/common/utils/metrics.h>

namespace similia {

using std::chrono::steady_clock;

using grpc::Status;
using grpc::ServerContext;

using common_utils::Timer;
using proto::CompressedElements;

Status InvertedMultiIndexService::Add(ServerContext* context,
                                      const proto::MultiIndexAddRequest* request,
                                      proto::MultiIndexAddResponse* response) {
  LOG(INFO) << "Add Request...";
  Timer time_add("inverted_multi_index_service.add.processing");
  steady_clock::time_point before_add = steady_clock::now();

  CompressedElements compressed_residual;
  compressed_residual.add_id(request->image_id());
  *compressed_residual.add_compressing_clusters_id() = request->compressing_ids();
  inverted_multi_index_->AddResidualCompressedToCluster(request->indexing_ids().id(0), request->indexing_ids().id(1),
                                                        compressed_residual);

  int processing_time = time_add.Stop();
  response->set_processing_time_ms(processing_time);
  steady_clock::time_point after_add = steady_clock::now();
  LOG(INFO) << "Add took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_add-before_add).count()
      << " us.";

  return Status::OK;
}

Status InvertedMultiIndexService::Get(ServerContext* context,
                                      const proto::MultiIndexGetRequest* request,
                                      proto::MultiIndexGetResponse* response) {
  LOG(INFO) << "Get Request...";
  Timer time_get("inverted_multi_index_service.get.processing");
  steady_clock::time_point before_get = steady_clock::now();

  int count = 0;
  response->set_compressed_elements(inverted_multi_index_->GetResidualsInCluster(request->indexing_ids().id(0),
                                                                                 request->indexing_ids().id(1),
                                                                                 &count));

  int processing_time = time_get.Stop();
  response->set_processing_time_ms(processing_time);
  steady_clock::time_point after_get = steady_clock::now();
  LOG(INFO) << "Get " << count << " elements took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_get-before_get).count()
      << " us.";

  return Status::OK;
}

Status InvertedMultiIndexService::MultiGet(ServerContext* context,
                                           const proto::MultiIndexMultiGetRequest* request,
                                           proto::MultiIndexMultiGetResponse* response) {
  LOG(INFO) << "MultiGet Request...";
  Timer time_get("inverted_multi_index_service.multi_get.processing");

  int num_images = 0;
  int time_get_only_ms = 0;
  for (const auto& it : request->indexing_ids()) {
    int count = 0;
    steady_clock::time_point before_get = steady_clock::now();
    response->add_compressed_elements(inverted_multi_index_->GetResidualsInCluster(it.id(0), it.id(1), &count));
    int time_get = std::chrono::duration_cast<std::chrono::microseconds>(
        steady_clock::now()-before_get).count();
    VLOG(1) << "Get " << count << " elements took: "
        << time_get
        << " us.";
    time_get_only_ms += time_get;
    num_images += count;
    if (num_images >= request->count_limit()) break;
  }

  time_get_only_ms /= 1000;
  int processing_time = time_get.Stop();
  response->set_processing_time_ms(processing_time);
  LOG(INFO) << "MultiGet " << response->compressed_elements_size() << " clusters with " << num_images <<
      " elements took: " << processing_time << " ms.";
  LOG(INFO) << "SumOfGetTime: " << time_get_only_ms << " ms";

  return Status::OK;
}

Status InvertedMultiIndexService::MultiCount(ServerContext* context,
                                             const proto::MultiIndexMultiCountRequest* request,
                                             proto::MultiIndexMultiCountResponse* response) {
  LOG(INFO) << "MultiCount Request...";
  Timer time_get("inverted_multi_index_service.multi_count.processing");

  int num_images = 0;
  for (const auto& it : request->indexing_ids()) {
    steady_clock::time_point before_getcount = steady_clock::now();
    num_images = inverted_multi_index_->GetCountForCluster(it.id(0), it.id(1));
    response->add_count(num_images);
    steady_clock::time_point after_getcount = steady_clock::now();
    VLOG(2) << "Get count " << num_images << " images took: "
        << std::chrono::duration_cast<std::chrono::microseconds>(after_getcount-before_getcount).count()
        << " us.";
  }

  int processing_time = time_get.Stop();
  response->set_processing_time_ms(processing_time);
  LOG(INFO) << "MultiCount " << response->count_size() << " counts took: " << processing_time << " ms.";

  return Status::OK;
}

Status InvertedMultiIndexService::MultiCountAtLastStartup(ServerContext* context,
                                                          const proto::MultiIndexMultiCountRequest* request,
                                                          proto::MultiIndexMultiCountResponse* response) {
  LOG(INFO) << "MultiCountFromCache Request...";
  Timer time_get("inverted_multi_index_service.multi_count.processing");

  int num_images = 0;
  for (const auto& it : request->indexing_ids()) {
    steady_clock::time_point before_getcount = steady_clock::now();
    num_images = inverted_multi_index_->GetCountAtLastStartup(it.id(0), it.id(1));
    response->add_count(num_images);
    steady_clock::time_point after_getcount = steady_clock::now();
    VLOG(2) << "Get count " << num_images << " images took: "
        << std::chrono::duration_cast<std::chrono::microseconds>(after_getcount-before_getcount).count()
        << " us.";
  }

  int processing_time = time_get.Stop();
  response->set_processing_time_ms(processing_time);
  LOG(INFO) << "MultiCount " << response->count_size() << " counts took: " << processing_time << " ms.";

  return Status::OK;
}

Status InvertedMultiIndexService::MultiAdd(ServerContext* context,
                                           const proto::MultiIndexMultiAddRequest* request,
                                           proto::MultiIndexMultiAddResponse* response) {
  LOG(INFO) << "MultiAdd Request with " << request->multi_index_add_request_size() << " elements ...";
  Timer time_add("inverted_multi_index_service.multi_add.processing");

  for (const auto& add_request : request->multi_index_add_request()) {
    CompressedElements compressed_residual;
    compressed_residual.add_id(add_request.image_id());
    *compressed_residual.add_compressing_clusters_id() = add_request.compressing_ids();
    inverted_multi_index_->AddResidualCompressedToCluster(add_request.indexing_ids().id(0),
                                                          add_request.indexing_ids().id(1), compressed_residual);
  }

  int processing_time = time_add.Stop();
  response->set_processing_time_ms(processing_time);
  LOG(INFO) << "MultiAdd with " << request->multi_index_add_request_size() << " additions took: " << processing_time
      << " ms.";

  return Status::OK;
}

}  // namespace similia
