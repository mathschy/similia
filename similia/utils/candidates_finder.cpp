#include <similia/utils/candidates_finder.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <set>
#include <utility>

#include <glog/logging.h>

#include <similia/common/utils/metrics.h>
#include <similia/proto/compressed_elements_generated.h>
#include <similia/proto/compressing_ids_generated.h>

namespace similia {
using std::chrono::steady_clock;

using common_utils::Timer;
using proto::IndexingClustersIds;
using proto::MultiIndexMultiGetRequest;
using proto::MultiIndexMultiGetResponse;

// that's 10000 clusters. should be enough for 10000 candidates.
static const int kNumClosestIndexingSubClusters = 100;

typedef Eigen::Matrix<bool, kNumClosestIndexingSubClusters, kNumClosestIndexingSubClusters> MatrixNCISCb;

CandidatesFinder::CandidatesFinder(const std::string& path_to_indexing_clusters_1_features_file,
                                   const std::string& path_to_indexing_clusters_2_features_file,
                                   std::unique_ptr<proto::InvertedMultiIndex::Stub> inverted_multi_index_client)
    : indexing_clusters_1_(path_to_indexing_clusters_1_features_file, kNumIndexingClustersPerDimensionDivision, 
                           kIndexingSubfeaturesDimensions, false, false),
      indexing_clusters_2_(path_to_indexing_clusters_2_features_file, kNumIndexingClustersPerDimensionDivision, 
                           kIndexingSubfeaturesDimensions, false, false),
      inverted_multi_index_client_(std::move(inverted_multi_index_client)) {

}

// for convenience.
IndexedValue BuildIndexedValue(int idx1, int idx2, float value) {
  IndexedValue indexed_value;
  indexed_value.indexes = {idx1, idx2};
  indexed_value.value = value;
  return indexed_value;
}

// implementation of the multisequence algorithm from The Inverted MultiIndex paper.
// http://download.yandex.ru/company/cvpr2012.pdf
// s1 and s1 are ordered by increasing order.
// return a list of indexes i,j that sort s1[i]+s2[j] in increasing order.
std::vector<std::pair<int, int>> MultiSequenceAlgorithm(const std::vector<float>& s1, const std::vector<float>& s2) {
  int idx1 = 0;
  int idx2 = 0;
  MatrixNCISCb selected = MatrixNCISCb::Constant(false);
  std::set<IndexedValue, IndexedValueComparator> queue_set;
  std::vector<std::pair<int, int>> ordered_indexes;  // we return this
  ordered_indexes.push_back({idx1, idx2});
  selected(idx1, idx2) = true;
  queue_set.insert(BuildIndexedValue(idx1 + 1, idx2, s1[idx1 + 1] + s2[idx2]));
  queue_set.insert(BuildIndexedValue(idx1, idx2 + 1, s1[idx1] + s2[idx2 + 1]));
  int max_number_selected = kNumClosestIndexingSubClusters * kNumClosestIndexingSubClusters;
  ordered_indexes.reserve(max_number_selected);
  IndexedValue min_element;
  while (ordered_indexes.size() < max_number_selected) { // for now we reorder everything
    // look for the min in the queue
    // the min in the queue is placed at the beginning
    min_element = *queue_set.begin();
    // pop this element from queue into selected
    ordered_indexes.push_back(min_element.indexes);
    selected(min_element.indexes.first, min_element.indexes.second) = true;
    queue_set.erase(queue_set.begin());  // direct access by iterator
    // add elements to the queue
    idx1 = min_element.indexes.first;
    idx2 = min_element.indexes.second;
    // see if we add idx1+1, idx2. we add it if it's in bounds, and if {idx1+1, idx2-1} is selected (or if idx2 = 0)
    if (idx1 + 1 < s1.size()) {
      if (idx2 == 0) {
        queue_set.insert(BuildIndexedValue(idx1 + 1, idx2, s1[idx1 + 1] + s2[idx2]));
      } else if (selected(idx1 + 1, idx2 - 1)) {
        queue_set.insert(BuildIndexedValue(idx1 + 1, idx2, s1[idx1 + 1] + s2[idx2]));
      }
    }
    // see if we add idx1, idx2+1. we add it if it's in bounds, and if {idx1-1, idx2+1} is selected (or if idx1 = 0)
    if (idx2 + 1 < s2.size()) {
      if (idx1 == 0) {
        queue_set.insert(BuildIndexedValue(idx1, idx2 + 1, s1[idx1] + s2[idx2 + 1]));
      } else if (selected(idx1 - 1, idx2 + 1)) {
        queue_set.insert(BuildIndexedValue(idx1, idx2 + 1, s1[idx1] + s2[idx2 + 1]));
      }
    }
  }

  return ordered_indexes;
}

std::vector<Candidate> CandidatesFinder::GetCandidates(
    const std::vector<float>& features, int list_length,
    const grpc::ServerContext& server_context) const {
  std::vector<float> features1(features.begin(), features.begin() + kIndexingSubfeaturesDimensions);
  std::vector<float> features2(features.begin() + kIndexingSubfeaturesDimensions, features.end());
  VLOG(1) << "features1.size() = " << features1.size();
  VLOG(1) << "features2.size() = " << features2.size();

  std::vector<IdAndComputation> clusters1 = indexing_clusters_1_.GetNearestNeighbors(features1,
                                                                                     kNumClosestIndexingSubClusters);
  std::vector<IdAndComputation> clusters2 = indexing_clusters_2_.GetNearestNeighbors(features2,
                                                                                     kNumClosestIndexingSubClusters);

  std::vector<float> squared_distances1(kNumClosestIndexingSubClusters);
  for (int i = 0; i < kNumClosestIndexingSubClusters; ++i) {
    squared_distances1[i] = clusters1[i].squared_distance;
  }  // ordered by increasing euclidean distance
  std::vector<float> squared_distances2(kNumClosestIndexingSubClusters);
  for (int i = 0; i < kNumClosestIndexingSubClusters; ++i) {
    squared_distances2[i] = clusters2[i].squared_distance;
  }  // ordered by increasing euclidean distance

  LOG(INFO) << "performing multisequence algorithm...";
  Timer timer_multisequence("similia.images.multisequence");
  std::vector<std::pair<int, int>> indexes = MultiSequenceAlgorithm(squared_distances1, squared_distances2);
  LOG(INFO) << "multisequence algorithm took : " << timer_multisequence.Stop() << " ms";
  VLOG(1) << "size of s1 : " << squared_distances1.size();
  VLOG(1) << "size of s2 : " << squared_distances2.size();
  VLOG(1) << "size of indexes : " << indexes.size();

  // We query all the clusters we have calculated the distance to.
  // We set a limit on the number of images and get back only the minimum number of clusters to reach this limit.
  MultiIndexMultiGetRequest multi_get_request;
  multi_get_request.set_count_limit(list_length);
  for (int i = 0; i < kNumClosestIndexingSubClusters * kNumClosestIndexingSubClusters; ++i) {
    IndexingClustersIds* indexing_clusters_ids = multi_get_request.add_indexing_ids();
    indexing_clusters_ids->add_id(std::stoi(clusters1[indexes[i].first].id));
    indexing_clusters_ids->add_id(std::stoi(clusters2[indexes[i].second].id));
  }
  MultiIndexMultiGetResponse multi_get_response;
  std::unique_ptr<grpc::ClientContext> context = grpc::ClientContext::FromServerContext(server_context);
  steady_clock::time_point before_get = steady_clock::now();
  grpc::Status status = inverted_multi_index_client_->MultiGet(context.get(), multi_get_request, &multi_get_response);
  steady_clock::time_point after_get = steady_clock::now();
  LOG(INFO) << "MultiGet took: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(after_get - before_get).count()
            << " ms.";

  if (status.ok()) {
    std::vector<Candidate> candidates;
    candidates.reserve(list_length);
    Candidate candidate;
    int count_images = 0;
    for (int i = 0; i < multi_get_response.compressed_elements_size(); ++i) {
      const fbs::CompressedElements* compressed_elements = fbs::GetCompressedElements(
          reinterpret_cast<const uint8_t*>(multi_get_response.compressed_elements(i).data()));
      if (compressed_elements->id()->size() > 0) {
        VLOG(1) << "visiting the " << i << " cluster: " << "num_images in cluster ("
            << multi_get_request.indexing_ids(i).id(0) << ", " << multi_get_request.indexing_ids(i).id(1) << ") : "
            << compressed_elements->id()->size();
        for (int j = 0; j < compressed_elements->id()->size(); ++j) {
          candidate.image_id = std::string(compressed_elements->id()->Get(j)->data(),
                                           compressed_elements->id()->Get(j)->size());
          candidate.dot_product = clusters1[indexes[i].first].dot_product + clusters2[indexes[i].second].dot_product;
          const fbs::CompressingIds* compressing_ids =
              compressed_elements->compressing_ids()->Get(j)->compressing_ids_nested_root();
          for (int i_fd = 0; i_fd < kCompressingDimensionDivision; ++i_fd) {
            candidate.compressing_clusters_ids[i_fd] = compressing_ids->id()->Get(i_fd);
          }
          candidate.indexing_clusters_ids[0] = multi_get_request.indexing_ids(i).id(0);
          candidate.indexing_clusters_ids[1] = multi_get_request.indexing_ids(i).id(1);
          candidates.push_back(candidate);
          ++count_images;
          VLOG(2) << "adding " << count_images << "th candidate: " << candidate.image_id << " with dotproduct: "
              << candidate.dot_product;
        }
      } else {
        // There was no image in cluster. do nothing.
      }
    }
    float average_images_per_cluster = static_cast<float>(count_images) / multi_get_response.compressed_elements_size();

    LOG(INFO) << "Found " << count_images << " images in " << multi_get_response.compressed_elements_size()
        << " clusters. That's " << average_images_per_cluster << " images per cluster on average.";
    return candidates;
  } else {
    LOG(ERROR) << "MultiGet error: " << status.error_code() << " : " << status.error_message();
    return {};
  }


}

}  // namespace similia
