#include "similia/utils/multi_product_quantizer_service.h"

#include <chrono>
#include <sstream>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include "similia/common/utils/metrics.h"
#include "similia/utils/features_utils.h"


namespace similia {

using std::chrono::steady_clock;

using grpc::Status;
using grpc::StatusCode;
using grpc::ServerContext;

using common_utils::Timer;
using proto::IndexingClustersIds;
using proto::CompressingClustersIds;


Status MultiProductQuantizerService::Quantize(ServerContext* context,
                                              const proto::QuantizationRequest* request,
                                              proto::QuantizationResponse* response) {
  if (!request->has_features()) {
    const std::string error_message = "There are no features in the request for id: " + request->image_id();
    LOG(ERROR) << error_message;
    return Status(StatusCode::INVALID_ARGUMENT, error_message);
  }
  if (request->features().element_size() != kFeatureDimensions) {
    std::ostringstream error_message_stream;
    error_message_stream << "Features have " << request->features().element_size() << " dimensions instead of " <<
        kFeatureDimensions << " required.";
    const std::string error_message = error_message_stream.str();
    LOG(ERROR) << error_message;
    return Status(StatusCode::INVALID_ARGUMENT, error_message);
  }
  Timer timer("similia.images.quantization");
  LOG(INFO) << "quantizing features of image_id: " << request->image_id();

  steady_clock::time_point before_f_creation = steady_clock::now();
  // store features in eigen vector
  Eigen::RowVectorXf f = Eigen::RowVectorXf::Map(&(request->features().element().Get(0)), kFeatureDimensions);

  Timer timer_normalization("similia.images.normalization");
  f.normalize();
  VLOG(1) << "elapsed_time for normalization: " << timer_normalization.Stop() << " ms";

  f = f * indexing_rotation_matrix_;
  steady_clock::time_point after_f_creation = steady_clock::now();
  VLOG(1) << "f vector creation took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_f_creation-before_f_creation).count()
      << " us.";
  // for convenience
  Eigen::MatrixXf::Index index;
  Eigen::VectorXf f_seg;
  // Get nearest indexing cluster.
  IndexingClustersIds indexing_clusters_ids;
  for (int i = 0; i < kIndexingDimensionDivision; ++i) {
    f_seg = f.segment((kIndexingSubfeaturesDimensions) * i,
                      (kIndexingSubfeaturesDimensions));
    ((indexing_clusters_norms_[i] - 2 * indexing_clusters_features_[i] * f_seg).array() + f_seg.squaredNorm())
        .minCoeff(&index);
    indexing_clusters_ids.add_id((int)index);  // index is small enough for int storage.
    CHECK_EQ((int)index, indexing_clusters_ids.id(i));
  }
  steady_clock::time_point after_min_coeff2 = steady_clock::now();
  VLOG(1) << "min coeff2 took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_min_coeff2-after_f_creation).count()
      << " us.";
  for (int i = 0; i < kIndexingDimensionDivision; ++i) {
    // substract the nearest cluster features to the features (compute residual).
    f.segment((kIndexingSubfeaturesDimensions) * i,
              (kIndexingSubfeaturesDimensions)) -= indexing_clusters_features_[i].row(indexing_clusters_ids.id(i));
  }
  f = f * compressing_rotation_matrix_;
  steady_clock::time_point after_substraction = steady_clock::now();
  VLOG(1) << "substractions took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_substraction-after_min_coeff2).count()
      << " us.";

  *response->mutable_indexing_ids() = indexing_clusters_ids;
  steady_clock::time_point after_indexing_assignments = steady_clock::now();
  VLOG(1) << "indexing assignments took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_indexing_assignments-after_substraction).count()
      << " us.";
  // Get nearest compressing cluster. f vector now contains residuals.
  CompressingClustersIds compressing_clusters_ids;
  for (int i = 0; i < kCompressingDimensionDivision; ++i) {
    f_seg = f.segment((kCompressingSubfeaturesDimensions) * i,
                      (kCompressingSubfeaturesDimensions));
    ((compressing_clusters_norms_[i] - 2 * compressing_clusters_features_[i] * f_seg).array() + f_seg.squaredNorm())
        .minCoeff(&index);
    compressing_clusters_ids.add_id((int)index);  // index is small enough for int storage.
  }
  steady_clock::time_point after_min_compress = steady_clock::now();
  VLOG(1) << "min compress took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_min_compress-after_indexing_assignments).count()
      << " us.";
  *response->mutable_compressing_ids() = compressing_clusters_ids;
  steady_clock::time_point after_compress_assignments = steady_clock::now();
  VLOG(1) << "compressing assignments took: "
      << std::chrono::duration_cast<std::chrono::microseconds>(after_compress_assignments-after_min_compress).count()
      << " us.";
  int elapsed_time = timer.Stop();
  LOG(INFO) << "elapsed_time: " << elapsed_time << " ms";
  response->set_processing_time_ms(elapsed_time);

  return Status::OK;
}

}  // namespace similia
